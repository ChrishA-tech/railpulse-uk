"""Guardrailed LangGraph NRCoT / TOC Delay Repay Copilot with hybrid BM25+TF-IDF retrieval and citation verification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Curated UK Rail regulatory & TOC Passenger Charter knowledge base
NRCOT_AND_TOC_CORPUS: Final[list[dict[str, str]]] = [
    {
        "doc_id": "NRCoT-14.2",
        "source": "National Rail Conditions of Travel (Condition 14.2)",
        "topic": "split_ticketing",
        "text": (
            "Condition 14.2 Split Ticketing Validity: A passenger may use two or more tickets for one "
            "journey as long as the train calls at every station where one ticket ends and the next "
            "begins (unless using a zonal or season ticket under Condition 14.3). If the train does "
            "not call at the split station, the split ticket combination is invalid."
        ),
    },
    {
        "doc_id": "NRCoT-14.3",
        "source": "National Rail Conditions of Travel (Condition 14.3)",
        "topic": "split_ticketing",
        "text": (
            "Condition 14.3 Zonal & Season Ticket Split Exception: If one of the tickets is a Season "
            "Ticket, Travelcard, or zonal ticket covering the relevant boundary, the train does not "
            "need to stop at the boundary station provided both tickets together cover the entire route."
        ),
    },
    {
        "doc_id": "NRCoT-28.1",
        "source": "National Rail Conditions of Travel (Condition 28.1 & ORR Guidance)",
        "topic": "delay_repay_general",
        "text": (
            "Condition 28 Delay Repay & Through-Journey Split-Ticket Protection: Compensation is "
            "calculated on the total delay to the passenger's end-to-end journey against the full "
            "combination of valid split tickets when minimum connection times are respected. Claims "
            "must be submitted to the Train Operating Company (TOC) responsible for the delay within "
            "28 days of completing the journey."
        ),
    },
    {
        "doc_id": "TOC-AVANTI-DR15",
        "source": "Avanti West Coast Passenger Charter (Delay Repay 15)",
        "topic": "toc_charter",
        "text": (
            "Avanti West Coast Delay Repay 15-minute rule: 15 to 29 minutes delay = 25% of single "
            "ticket (12.5% of return); 30 to 59 minutes delay = 50% of single ticket (25% of return); "
            "60 to 119 minutes delay = 100% of single ticket (50% of return); 120+ minutes delay = "
            "100% of single or return ticket."
        ),
    },
    {
        "doc_id": "TOC-NORTHERN-DR15",
        "source": "Northern Trains Passenger Charter (Delay Repay 15)",
        "topic": "toc_charter",
        "text": (
            "Northern Trains Delay Repay 15-minute rule: 15 to 29 minutes delay = 25% of single "
            "fare (12.5% of return); 30 to 59 minutes delay = 50% of single fare (25% of return); "
            "60 to 119 minutes delay = 100% of single fare (50% of return); 120+ minutes delay = "
            "100% of single or return fare."
        ),
    },
    {
        "doc_id": "TOC-TPE-DR15",
        "source": "TransPennine Express Passenger Charter (Delay Repay 15)",
        "topic": "toc_charter",
        "text": (
            "TransPennine Express (TPE) Delay Repay 15-minute rule: 15 to 29 minutes delay = 25% of "
            "single fare (12.5% of return); 30 to 59 minutes delay = 50% of single fare (25% of return); "
            "60 to 119 minutes delay = 100% of single fare (50% of return); 120+ minutes delay = "
            "100% of single or return fare."
        ),
    },
    {
        "doc_id": "TOC-LNER-DR30",
        "source": "LNER Passenger Charter (Delay Repay 30)",
        "topic": "toc_charter",
        "text": (
            "London North Eastern Railway (LNER) Delay Repay 30-minute rule: LNER does not pay Delay "
            "Repay for delays under 30 minutes (0% for 15-29 minutes). 30 to 59 minutes delay = 50% of "
            "single fare (25% of return); 60 to 119 minutes delay = 100% of single fare (50% of return); "
            "120+ minutes delay = 100% of single or return fare."
        ),
    },
    {
        "doc_id": "ORR-DISRUPTION-32",
        "source": "ORR & NRCoT Condition 32 (Severe Disruption)",
        "topic": "disruption",
        "text": (
            "NRCoT Condition 32 Disruption & Ticket Acceptance: During severe weather or signal "
            "failures, TOCs must arrange alternative transport or ticket acceptance on other routes "
            "and may not refuse Delay Repay for extraordinary weather events (Network Rail code X8)."
        ),
    },
]

TOC_ALIASES: Final[dict[str, str]] = {
    "avanti": "Avanti West Coast",
    "avanti west coast": "Avanti West Coast",
    "wcml": "Avanti West Coast",
    "northern": "Northern",
    "northern trains": "Northern",
    "lner": "LNER",
    "ecml": "LNER",
    "london north eastern railway": "LNER",
    "tpe": "TransPennine Express",
    "transpennine": "TransPennine Express",
    "transpennine express": "TransPennine Express",
}

# Minimum statutory Delay Repay threshold in minutes by TOC
TOC_THRESHOLD_MIN: Final[dict[str, int]] = {
    "Avanti West Coast": 15,
    "Northern": 15,
    "TransPennine Express": 15,
    "LNER": 30,
}

TOC_DOC_MAP: Final[dict[str, str]] = {
    "Avanti West Coast": "TOC-AVANTI-DR15",
    "Northern": "TOC-NORTHERN-DR15",
    "TransPennine Express": "TOC-TPE-DR15",
    "LNER": "TOC-LNER-DR30",
}

PROMPT_INJECTION_PATTERNS: Final[tuple[str, ...]] = (
    "ignore previous",
    "ignore nrcot",
    "override statutory",
    "force 100%",
    "bypass citation",
    "system prompt",
)


def normalise_toc_name(toc: str) -> str:
    """Normalise TOC names/abbreviations to canonical operator names."""
    key = toc.strip().lower()
    if key in TOC_ALIASES:
        return TOC_ALIASES[key]
    for alias, canonical in TOC_ALIASES.items():
        if alias in key:
            return canonical
    return "Avanti West Coast" if not key else toc.strip()


def calculate_statutory_delay_repay(
    toc: str,
    delay_minutes: float,
    ticket_type: str = "single",
    is_split_ticket: bool = False,
    train_calls_at_split_station: bool = True,
    has_zonal_or_season_split: bool = False,
    ticket_price_gbp: float | None = None,
) -> dict[str, Any]:
    """Deterministic statutory UK Delay Repay & NRCoT split-ticket calculator."""
    canonical_toc = normalise_toc_name(toc)
    threshold = TOC_THRESHOLD_MIN.get(canonical_toc, 30)
    is_return = ticket_type.strip().lower() == "return"

    split_valid = True
    split_reason = "Single through-ticket (Condition 14 not triggered)."
    if is_split_ticket:
        if train_calls_at_split_station:
            split_valid = True
            split_reason = "Split ticket valid under NRCoT Condition 14.2 (train calls at split station)."
        elif has_zonal_or_season_split:
            split_valid = True
            split_reason = "Split ticket valid under NRCoT Condition 14.3 (zonal/season ticket exception)."
        else:
            split_valid = False
            split_reason = (
                "Split ticket INVALID under NRCoT Condition 14.2: train does not call at the split "
                "station and no Condition 14.3 zonal/season exception applies."
            )

    if not split_valid:
        pct = 0.0
        band = "INVALID_SPLIT_TICKET"
    elif delay_minutes < threshold:
        pct = 0.0
        band = f"BELOW_{threshold}M_THRESHOLD"
    elif 15.0 <= delay_minutes < 30.0 and threshold == 15:
        pct = 12.5 if is_return else 25.0
        band = "15_TO_29_MIN"
    elif 30.0 <= delay_minutes < 60.0:
        pct = 25.0 if is_return else 50.0
        band = "30_TO_59_MIN"
    elif 60.0 <= delay_minutes < 120.0:
        pct = 50.0 if is_return else 100.0
        band = "60_TO_119_MIN"
    else:
        pct = 100.0
        band = "120_PLUS_MIN"

    refund_gbp = round(ticket_price_gbp * (pct / 100.0), 2) if ticket_price_gbp is not None else None
    return {
        "toc": canonical_toc,
        "threshold_minutes": threshold,
        "delay_minutes": float(delay_minutes),
        "ticket_type": "return" if is_return else "single",
        "compensation_pct": pct,
        "delay_band": band,
        "split_ticket_valid": split_valid,
        "split_ticket_reason": split_reason,
        "estimated_refund_gbp": refund_gbp,
    }


class HybridRegulationRetriever:
    """Hybrid BM25 + TF-IDF cosine retriever over NRCoT and TOC Passenger Charters."""

    def __init__(self, corpus: list[dict[str, str]] | None = None) -> None:
        self.corpus = corpus or list(NRCOT_AND_TOC_CORPUS)
        self._tokenized = [self._tokenize(f"{d['doc_id']} {d['source']} {d['text']}") for d in self.corpus]
        self._bm25 = BM25Okapi(self._tokenized)
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        texts = [f"{d['source']} {d['text']}" for d in self.corpus]
        self._tfidf_matrix = self._tfidf.fit_transform(texts)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """Retrieve top_k clauses ranked by normalised 0.5*BM25 + 0.5*TF-IDF hybrid score."""
        q_tokens = self._tokenize(query)
        bm25_scores = np.asarray(self._bm25.get_scores(q_tokens), dtype=float)
        if bm25_scores.max() > 0:
            bm25_norm = bm25_scores / bm25_scores.max()
        else:
            bm25_norm = bm25_scores

        q_vec = self._tfidf.transform([query])
        tfidf_scores = cosine_similarity(q_vec, self._tfidf_matrix).ravel()
        if tfidf_scores.max() > 0:
            tfidf_norm = tfidf_scores / tfidf_scores.max()
        else:
            tfidf_norm = tfidf_scores

        hybrid = 0.5 * bm25_norm + 0.5 * tfidf_norm
        order = np.argsort(hybrid)[::-1][:top_k]

        hits: list[dict[str, Any]] = []
        for idx in order:
            doc = self.corpus[int(idx)]
            hits.append(
                {
                    "doc_id": doc["doc_id"],
                    "source": doc["source"],
                    "topic": doc["topic"],
                    "text": doc["text"],
                    "hybrid_score": round(float(hybrid[int(idx)]), 4),
                    "bm25_score": round(float(bm25_norm[int(idx)]), 4),
                    "tfidf_score": round(float(tfidf_norm[int(idx)]), 4),
                }
            )
        return hits


class CopilotState(TypedDict, total=False):
    """LangGraph workflow state for NRCoT / Delay Repay auditing."""

    query: str
    toc: str
    delay_minutes: float
    ticket_type: str
    is_split_ticket: bool
    train_calls_at_split_station: bool
    has_zonal_or_season_split: bool
    ticket_price_gbp: float | None
    guardrail_flagged: bool
    guardrail_reason: str
    retrieved_docs: list[dict[str, Any]]
    draft_answer: str
    draft_citations: list[str]
    draft_pct: float
    verification_passed: bool
    verification_notes: str
    used_statutory_fallback: bool
    final_result: dict[str, Any]


class DelayRepayCopilot:
    """Guardrailed LangGraph copilot auditing NRCoT split-ticket and TOC Delay Repay claims."""

    def __init__(self, retriever: HybridRegulationRetriever | None = None) -> None:
        self.retriever = retriever or HybridRegulationRetriever()
        self.graph = self._build_graph()

    def _guardrail_and_retrieve_node(self, state: CopilotState) -> CopilotState:
        query = state.get("query", "")
        toc = normalise_toc_name(state.get("toc", "Avanti West Coast"))
        q_lower = query.lower()

        flagged = any(pat in q_lower for pat in PROMPT_INJECTION_PATTERNS)
        reason = (
            "Prompt-override attempt detected; enforcing statutory NRCoT and TOC rules."
            if flagged
            else ""
        )

        augmented_query = (
            f"{query} {toc} delay repay {state.get('delay_minutes', 0)} minutes "
            f"{'split ticket condition 14.2' if state.get('is_split_ticket') else 'condition 28'}"
        )
        retrieved = self.retriever.search(augmented_query, top_k=4)

        # Ensure the primary TOC charter clause & NRCoT-28.1 / NRCoT-14.2 are present if relevant
        existing_ids = {d["doc_id"] for d in retrieved}
        required_ids = [TOC_DOC_MAP.get(toc, "TOC-AVANTI-DR15"), "NRCoT-28.1"]
        if state.get("is_split_ticket"):
            required_ids.append("NRCoT-14.2")
            if state.get("has_zonal_or_season_split"):
                required_ids.append("NRCoT-14.3")

        corpus_by_id = {d["doc_id"]: d for d in self.retriever.corpus}
        for req_id in required_ids:
            if req_id not in existing_ids and req_id in corpus_by_id:
                doc = corpus_by_id[req_id]
                retrieved.append(
                    {
                        "doc_id": doc["doc_id"],
                        "source": doc["source"],
                        "topic": doc["topic"],
                        "text": doc["text"],
                        "hybrid_score": 0.5,
                        "bm25_score": 0.5,
                        "tfidf_score": 0.5,
                    }
                )
                existing_ids.add(req_id)

        return {
            "toc": toc,
            "guardrail_flagged": flagged,
            "guardrail_reason": reason,
            "retrieved_docs": retrieved,
        }

    def _draft_decision_node(self, state: CopilotState) -> CopilotState:
        calc = calculate_statutory_delay_repay(
            toc=state["toc"],
            delay_minutes=state.get("delay_minutes", 0.0),
            ticket_type=state.get("ticket_type", "single"),
            is_split_ticket=state.get("is_split_ticket", False),
            train_calls_at_split_station=state.get("train_calls_at_split_station", True),
            has_zonal_or_season_split=state.get("has_zonal_or_season_split", False),
            ticket_price_gbp=state.get("ticket_price_gbp"),
        )
        retrieved_ids = [d["doc_id"] for d in state.get("retrieved_docs", [])]
        toc_doc = TOC_DOC_MAP.get(calc["toc"], "TOC-AVANTI-DR15")
        citations = [cid for cid in (toc_doc, "NRCoT-28.1") if cid in retrieved_ids]
        if state.get("is_split_ticket") and "NRCoT-14.2" in retrieved_ids:
            citations.append("NRCoT-14.2")
        if state.get("has_zonal_or_season_split") and "NRCoT-14.3" in retrieved_ids:
            citations.append("NRCoT-14.3")

        cite_tags = " ".join(f"[{c}]" for c in citations)
        refund_part = (
            f" (£{calc['estimated_refund_gbp']:.2f} estimated refund)"
            if calc["estimated_refund_gbp"] is not None
            else ""
        )
        draft = (
            f"Operator: {calc['toc']} ({calc['threshold_minutes']}-min charter). "
            f"Delay: {calc['delay_minutes']:.0f} min on {calc['ticket_type']} ticket -> "
            f"{calc['compensation_pct']:.1f}% Delay Repay{refund_part}. "
            f"{calc['split_ticket_reason']} Citations: {cite_tags}"
        )
        return {
            "draft_answer": draft,
            "draft_citations": citations,
            "draft_pct": calc["compensation_pct"],
        }

    def _citation_verifier_judge_node(self, state: CopilotState) -> CopilotState:
        """Judge node verifying every cited clause is grounded in retrieval and matches statutory math."""
        retrieved_ids = {d["doc_id"] for d in state.get("retrieved_docs", [])}
        draft_citations = state.get("draft_citations", [])
        draft_answer = state.get("draft_answer", "")

        # Extract any [DOC-ID] tags from draft text
        tags_in_text = re.findall(r"\[([A-Za-z0-9.\-]+)\]", draft_answer)
        all_cited = set(draft_citations) | set(tags_in_text)

        statutory = calculate_statutory_delay_repay(
            toc=state["toc"],
            delay_minutes=state.get("delay_minutes", 0.0),
            ticket_type=state.get("ticket_type", "single"),
            is_split_ticket=state.get("is_split_ticket", False),
            train_calls_at_split_station=state.get("train_calls_at_split_station", True),
            has_zonal_or_season_split=state.get("has_zonal_or_season_split", False),
            ticket_price_gbp=state.get("ticket_price_gbp"),
        )

        ungrounded = all_cited - retrieved_ids
        has_citations = len(all_cited) > 0
        pct_matches = abs(float(state.get("draft_pct", -1.0)) - statutory["compensation_pct"]) < 1e-6

        if state.get("guardrail_flagged"):
            passed = False
            notes = "Guardrail triggered by adversarial override attempt; routing to deterministic statutory fallback."
        elif ungrounded:
            passed = False
            notes = f"Hallucinated citations detected ({sorted(ungrounded)}); routing to statutory fallback."
        elif not has_citations:
            passed = False
            notes = "No grounded NRCoT/TOC citations produced; routing to statutory fallback."
        elif not pct_matches:
            passed = False
            notes = "Draft percentage deviated from statutory rule; routing to statutory fallback."
        else:
            passed = True
            notes = f"Verified {len(all_cited)} grounded citation(s) against retrieved NRCoT/TOC corpus."

        return {
            "verification_passed": passed,
            "verification_notes": notes,
            "used_statutory_fallback": not passed,
            "final_result": {
                **statutory,
                "answer": draft_answer,
                "citations": sorted(all_cited),
                "verification_passed": passed,
                "verification_notes": notes,
                "used_statutory_fallback": not passed,
                "guardrail_flagged": bool(state.get("guardrail_flagged", False)),
                "retrieved_docs": state.get("retrieved_docs", []),
            },
        }

    def _statutory_fallback_node(self, state: CopilotState) -> CopilotState:
        """Deterministic fallback enforcing statutory NRCoT & TOC Delay Repay calculation."""
        statutory = calculate_statutory_delay_repay(
            toc=state["toc"],
            delay_minutes=state.get("delay_minutes", 0.0),
            ticket_type=state.get("ticket_type", "single"),
            is_split_ticket=state.get("is_split_ticket", False),
            train_calls_at_split_station=state.get("train_calls_at_split_station", True),
            has_zonal_or_season_split=state.get("has_zonal_or_season_split", False),
            ticket_price_gbp=state.get("ticket_price_gbp"),
        )
        toc_doc = TOC_DOC_MAP.get(statutory["toc"], "TOC-AVANTI-DR15")
        fallback_citations = [toc_doc, "NRCoT-28.1"]
        if state.get("is_split_ticket"):
            fallback_citations.append("NRCoT-14.2")
        if state.get("has_zonal_or_season_split"):
            fallback_citations.append("NRCoT-14.3")

        answer = (
            f"[Statutory Fallback] {statutory['toc']} ({statutory['threshold_minutes']}-min rule): "
            f"{statutory['delay_minutes']:.0f} min delay on {statutory['ticket_type']} fare qualifies for "
            f"{statutory['compensation_pct']:.1f}% Delay Repay. {statutory['split_ticket_reason']} "
            f"Citations: {' '.join(f'[{c}]' for c in fallback_citations)}"
        )
        return {
            "used_statutory_fallback": True,
            "final_result": {
                **statutory,
                "answer": answer,
                "citations": fallback_citations,
                "verification_passed": False,
                "verification_notes": state.get("verification_notes", "Statutory fallback executed."),
                "used_statutory_fallback": True,
                "guardrail_flagged": bool(state.get("guardrail_flagged", False)),
                "retrieved_docs": state.get("retrieved_docs", []),
            },
        }

    @staticmethod
    def _route_after_verification(state: CopilotState) -> str:
        return "verified" if state.get("verification_passed", False) else "fallback"

    def _build_graph(self):
        builder = StateGraph(CopilotState)
        builder.add_node("retrieve", self._guardrail_and_retrieve_node)
        builder.add_node("draft", self._draft_decision_node)
        builder.add_node("verify", self._citation_verifier_judge_node)
        builder.add_node("statutory_fallback", self._statutory_fallback_node)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "draft")
        builder.add_edge("draft", "verify")
        builder.add_conditional_edges(
            "verify",
            self._route_after_verification,
            {
                "verified": END,
                "fallback": "statutory_fallback",
            },
        )
        builder.add_edge("statutory_fallback", END)
        return builder.compile()

    def verify_external_draft(
        self,
        draft_answer: str,
        draft_pct: float,
        toc: str,
        delay_minutes: float,
        ticket_type: str = "single",
        is_split_ticket: bool = False,
        train_calls_at_split_station: bool = True,
        has_zonal_or_season_split: bool = False,
    ) -> dict[str, Any]:
        """Run the citation-verifier judge & statutory fallback on an arbitrary draft response."""
        state = self._guardrail_and_retrieve_node(
            {
                "query": draft_answer,
                "toc": toc,
                "delay_minutes": delay_minutes,
                "ticket_type": ticket_type,
                "is_split_ticket": is_split_ticket,
                "train_calls_at_split_station": train_calls_at_split_station,
                "has_zonal_or_season_split": has_zonal_or_season_split,
            }
        )
        state.update(
            {
                "delay_minutes": delay_minutes,
                "ticket_type": ticket_type,
                "is_split_ticket": is_split_ticket,
                "train_calls_at_split_station": train_calls_at_split_station,
                "has_zonal_or_season_split": has_zonal_or_season_split,
                "draft_answer": draft_answer,
                "draft_citations": re.findall(r"\[([A-Za-z0-9.\-]+)\]", draft_answer),
                "draft_pct": draft_pct,
            }
        )
        verified_state = self._citation_verifier_judge_node(state)
        if not verified_state["verification_passed"]:
            state.update(verified_state)
            fallback_state = self._statutory_fallback_node(state)
            return fallback_state["final_result"]
        return verified_state["final_result"]

    def audit_claim(
        self,
        query: str,
        toc: str,
        delay_minutes: float,
        ticket_type: str = "single",
        is_split_ticket: bool = False,
        train_calls_at_split_station: bool = True,
        has_zonal_or_season_split: bool = False,
        ticket_price_gbp: float | None = None,
    ) -> dict[str, Any]:
        """Execute the full LangGraph NRCoT Delay Repay auditing workflow."""
        initial_state: CopilotState = {
            "query": query,
            "toc": toc,
            "delay_minutes": float(delay_minutes),
            "ticket_type": ticket_type,
            "is_split_ticket": is_split_ticket,
            "train_calls_at_split_station": train_calls_at_split_station,
            "has_zonal_or_season_split": has_zonal_or_season_split,
            "ticket_price_gbp": ticket_price_gbp,
        }
        out = self.graph.invoke(initial_state)
        return out["final_result"]


@dataclass
class CopilotSummary:
    """Convenience wrapper for pipeline verification."""

    verified_claims: int
    fallback_triggered: int
