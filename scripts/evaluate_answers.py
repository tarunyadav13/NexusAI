import json
import logging
import os
import re
import time

from pathlib import Path
from dotenv import load_dotenv
from google import genai

from nexusai.agentic.graph import build_agentic_graph
from nexusai.agentic.llm import GeminiLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(ROOT_DIR / ".env")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

DATASET_PATH = (
    ROOT_DIR / "src/nexusai/evaluation/datasets/answer_eval.json"
)

RESULTS_DIR = (
    ROOT_DIR / "src/nexusai/evaluation/results"
)


def extract_json(text):

    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE
    )

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("Invalid evaluator JSON")

    return json.loads(match.group())


def get_evidence(output):

    documents = (
        output.get("graded_documents")
        or output.get("retrieved_documents")
        or []
    )

    evidence = []

    for doc in documents:

        if isinstance(doc, str):
            evidence.append(doc)

        elif isinstance(doc, dict):

            content = (
                doc.get("content")
                or doc.get("text")
                or doc.get("page_content")
                or ""
            )

            if content:
                evidence.append(content)

    return "\n\n".join(evidence)


def verify_citations(answer, citations):

    citation_ids = set()

    for citation in citations:

        if isinstance(citation, dict):
            citation_id = citation.get("id")

            if citation_id:
                citation_ids.add(str(citation_id))

        elif isinstance(citation, str):
            citation_ids.add(citation)

    answer_ids = set(
        re.findall(r"\[(S\d+)\]", answer)
    )

    valid = answer_ids.intersection(citation_ids)

    return {
        "citation_ids_in_answer": sorted(answer_ids),
        "available_citation_ids": sorted(citation_ids),
        "valid_citation_ids": sorted(valid),
        "citation_reference_precision": (
            len(valid) / len(answer_ids)
            if answer_ids else 0.0
        ),
        "has_citations": bool(answer_ids),
        "all_citations_resolved": (
            bool(answer_ids) and answer_ids.issubset(citation_ids)
        )
    }


def evaluate_with_judge(
    judge,
    query,
    reference,
    evidence,
    answer
):

    prompt = f"""
You are evaluating a Retrieval Augmented Generation system.

Question:
{query}

Reference Answer:
{reference}

Retrieved Evidence:
{evidence}

Generated Answer:
{answer}

Evaluate the generated answer.

Faithfulness:
Are the factual claims supported by the retrieved evidence?

Answer Relevance:
Does the answer correctly address the question and match
the essential information in the reference answer?

Use scores from 0 to 1.

Important:
- Do not reward unsupported claims.
- If evidence does not contain enough information, an answer
  acknowledging insufficient evidence can receive a high
  faithfulness score.
- Judge answer relevance separately from faithfulness.

Return ONLY JSON:

{{
    "faithfulness": 0.0,
    "answer_relevance": 0.0,
    "reason": "Explanation"
}}
"""

    response = judge.generate(prompt)

    result = extract_json(response)

    faithfulness = float(result["faithfulness"])
    relevance = float(result["answer_relevance"])

    if not 0 <= faithfulness <= 1:
        raise ValueError("Invalid faithfulness score")

    if not 0 <= relevance <= 1:
        raise ValueError("Invalid relevance score")

    return result


def estimate_tokens(client, text):

    try:

        response = client.models.count_tokens(
            model=MODEL,
            contents=text
        )

        return getattr(response, "total_tokens", None)

    except Exception as exc:

        logger.warning("Token counting failed: %s", exc)

        return None


def main():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    graph, retriever = build_agentic_graph()

    judge = GeminiLLM()

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    results = []

    try:

        for case in dataset:

            query = case["query"]

            logger.info("Evaluating: %s", query)

            start = time.perf_counter()

            output = graph.invoke({
                "original_query": query,
                "rewritten_query": "",
                "current_query": query,
                "retrieved_documents": [],
                "graded_documents": [],
                "evidence_score": 0.0,
                "evidence_sufficient": False,
                "retry_count": 0,
                "answer": "",
                "citations": [],
                "messages": []
            })

            execution_time = time.perf_counter() - start

            answer = output.get("answer", "")

            evidence = get_evidence(output)

            citations = output.get("citations", [])

            if not answer.strip():
                raise ValueError(f"Empty answer for: {query}")

            evaluation = evaluate_with_judge(
                judge,
                query,
                case["reference_answer"],
                evidence,
                answer
            )

            citation_result = verify_citations(
                answer,
                citations
            )

            token_input = estimate_tokens(
                client,
                query + "\n" + evidence
            )

            token_output = estimate_tokens(
                client,
                answer
            )

            result = {
                "query": query,
                "generated_answer": answer,
                "faithfulness": float(evaluation["faithfulness"]),
                "answer_relevance": float(evaluation["answer_relevance"]),
                "evaluation_reason": evaluation.get("reason", ""),
                "citations": citation_result,
                "citation_count": len(citations),
                "answer_latency_seconds": round(execution_time, 4),
                "input_tokens_estimated": token_input,
                "output_tokens_estimated": token_output,
                "evidence_score": output.get("evidence_score"),
                "evidence_sufficient": output.get("evidence_sufficient"),
                "retry_count": output.get("retry_count", 0)
            }

            results.append(result)

            print(json.dumps(result, indent=2))

    finally:

        retriever.close()
        client.close()

    metrics = [
        "faithfulness",
        "answer_relevance",
        "answer_latency_seconds"
    ]

    summary = {}

    for metric in metrics:

        values = [item[metric] for item in results]

        summary[metric] = round(
            sum(values) / len(values), 4
        ) if values else 0.0

    summary["citation_reference_precision"] = round(
        sum(
            item["citations"]["citation_reference_precision"]
            for item in results
        ) / len(results), 4
    ) if results else 0.0

    summary["total_input_tokens_estimated"] = sum(
        item["input_tokens_estimated"] or 0
        for item in results
    )

    summary["total_output_tokens_estimated"] = sum(
        item["output_tokens_estimated"] or 0
        for item in results
    )

    final_output = {
        "model": MODEL,
        "total_queries": len(results),
        "summary": summary,
        "individual_results": results
    }

    output_path = RESULTS_DIR / "answer_results.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(final_output, file, indent=4)

    print("\nNEXUSAI FINAL ANSWER EVALUATION")

    print(json.dumps(summary, indent=4))

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()