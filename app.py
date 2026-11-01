import os
import json
import math
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
import imageio.v2 as imageio
import streamlit as st


APP_TITLE = "Automated Knowledge Discovery Engine — Live Demo"
SEED = 20261101
OUTPUT_DIR = Path("output")
BASE_TIME = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()

TYPE_COLORS = {
    "Company": "#4C78A8",
    "Technology": "#F58518",
    "Product": "#54A24B",
    "Regulation": "#E45756",
    "ResearchGroup": "#B279A2",
    "Market": "#72B7B2",
}

RELATION_BY_TYPES = {
    ("Company", "Company"): "collaborates_with",
    ("Company", "Product"): "produces",
    ("Company", "Technology"): "uses",
    ("Company", "Regulation"): "regulated_by",
    ("Company", "ResearchGroup"): "collaborates_with",
    ("Company", "Market"): "serves",
    ("Technology", "Product"): "enables",
    ("ResearchGroup", "Technology"): "researches",
    ("Product", "Regulation"): "complies_with",
    ("Technology", "Technology"): "integrates_with",
    ("Product", "Product"): "interoperates_with",
    ("Market", "Regulation"): "governed_by",
}

ENTITY_RECORDS = [
    ("helios_robotics", "Helios Robotics", "Company", "Industrial robots, digital twins, predictive maintenance, factory automation."),
    ("northwind_logistics", "Northwind Logistics", "Company", "Logistics optimization, warehouse automation, fleet operations, industrial AI."),
    ("quantum_ledger", "Quantum Ledger", "Company", "Blockchain settlement, secure APIs, financial infrastructure, cyber compliance."),
    ("biosynth_labs", "BioSynth Labs", "Company", "Synthetic biology, CRISPR screening, genomic assays, healthcare compliance."),
    ("azuregrid_energy", "AzureGrid Energy", "Company", "Energy storage, solid-state batteries, edge inference, grid balance controllers."),
    ("meridian_health", "Meridian Health", "Company", "Healthcare IT, federated learning, edge inference, HIPAA compliant platforms."),
    ("orion_semiconductors", "Orion Semiconductors", "Company", "Semiconductor manufacturing, edge chips, predictive maintenance, security certification."),
    ("vega_analytics", "Vega Analytics", "Company", "Industrial analytics, federated learning, predictive maintenance, European AI markets."),

    ("federated_learning", "Federated Learning", "Technology", "Privacy preserving distributed machine learning, decentralized training, healthcare AI."),
    ("digital_twin", "Digital Twin", "Technology", "Simulation, robotics, factory modeling, real time operations, industrial automation."),
    ("solid_state_battery", "Solid State Battery", "Technology", "Energy storage, battery chemistry, grid resilience, electric vehicles."),
    ("crispr_screening", "CRISPR Screening", "Technology", "Genome editing, functional genomics, drug discovery, biological assays."),
    ("edge_inference", "Edge Inference", "Technology", "Low latency AI, embedded processors, factory floor, grid devices, medical devices."),
    ("blockchain_settlement", "Blockchain Settlement", "Technology", "Distributed ledgers, financial clearing, smart contracts, audit trails."),
    ("predictive_maintenance", "Predictive Maintenance", "Technology", "Sensor analytics, failure prediction, industrial operations, reliability engineering."),
    ("synthetic_biology", "Synthetic Biology", "Technology", "Engineered organisms, biomanufacturing, gene circuits, laboratory automation."),

    ("fleetops_suite", "FleetOps Suite", "Product", "Robot fleet management, digital twin operations, predictive maintenance dashboards."),
    ("genemap_assay", "GeneMap Assay", "Product", "CRISPR screening assay, synthetic biology workflow, genomic reporting."),
    ("gridbalance_controller", "GridBalance Controller", "Product", "Energy grid control, battery orchestration, edge inference telemetry."),
    ("secureledger_api", "SecureLedger API", "Product", "Blockchain settlement API, financial audit integration, cyber disclosure reporting."),
    ("meditrace_platform", "MediTrace Platform", "Product", "Healthcare traceability, federated learning, HIPAA compliant clinical operations."),

    ("eu_ai_act", "EU AI Act", "Regulation", "European artificial intelligence regulation, risk classification, conformity assessment."),
    ("hipaa", "HIPAA", "Regulation", "US healthcare privacy and security rule, protected health information, clinical data governance."),
    ("sec_cyber_disclosure", "SEC Cyber Disclosure", "Regulation", "Public company cybersecurity incident disclosure, material risk reporting."),
    ("iso_27001", "ISO 27001", "Regulation", "Information security management systems, certification, supplier security controls."),

    ("eth_zurich_rl_lab", "ETH Zurich RL Lab", "ResearchGroup", "Reinforcement learning research, federated learning, robotics, industrial AI."),
    ("stanford_bio_x", "Stanford Bio-X", "ResearchGroup", "Bioengineering research, CRISPR methods, synthetic biology, translational genomics."),
    ("mit_energy_initiative", "MIT Energy Initiative", "ResearchGroup", "Energy systems research, solid state batteries, grid modernization, storage."),

    ("eu_industrial_ai", "EU Industrial AI", "Market", "European industrial artificial intelligence, manufacturing automation, robotics software."),
    ("us_healthcare_it", "US Healthcare IT", "Market", "American healthcare information technology, clinical platforms, privacy compliance."),
    ("global_energy_storage", "Global Energy Storage", "Market", "Battery storage markets, grid services, renewable integration, utilities."),
]

OBSERVED_TRIPLES = [
    ("helios_robotics", "produces", "fleetops_suite", "helios_annual_report_2026.pdf", 120, 184, 0.96),
    ("helios_robotics", "uses", "digital_twin", "helios_tech_whitepaper.pdf", 305, 351, 0.93),
    ("helios_robotics", "uses", "predictive_maintenance", "helios_case_study.pdf", 88, 142, 0.91),
    ("helios_robotics", "collaborates_with", "eth_zurich_rl_lab", "eth_helios_press_release.pdf", 40, 112, 0.88),
    ("helios_robotics", "regulated_by", "eu_ai_act", "eu_ai_act_compliance_memo.pdf", 210, 268, 0.94),

    ("northwind_logistics", "uses", "predictive_maintenance", "northwind_ops_report.pdf", 150, 201, 0.89),
    ("northwind_logistics", "uses", "fleetops_suite", "northwind_vendor_review.pdf", 92, 145, 0.84),
    ("northwind_logistics", "serves", "eu_industrial_ai", "northwind_market_deck.pdf", 330, 376, 0.86),

    ("quantum_ledger", "produces", "secureledger_api", "quantum_product_page.html", 64, 119, 0.95),
    ("quantum_ledger", "uses", "blockchain_settlement", "quantum_architecture.pdf", 201, 260, 0.92),
    ("quantum_ledger", "regulated_by", "sec_cyber_disclosure", "quantum_legal_filing.pdf", 410, 468, 0.90),
    ("quantum_ledger", "collaborates_with", "vega_analytics", "quantum_vega_partnership.pdf", 75, 133, 0.87),

    ("biosynth_labs", "produces", "genemap_assay", "biosynth_product_insert.pdf", 55, 118, 0.96),
    ("biosynth_labs", "uses", "crispr_screening", "biosynth_methods.pdf", 220, 279, 0.94),
    ("biosynth_labs", "uses", "synthetic_biology", "biosynth_platform_overview.pdf", 130, 188, 0.91),
    ("biosynth_labs", "collaborates_with", "stanford_bio_x", "stanford_biosynth_agreement.pdf", 88, 149, 0.89),
    ("biosynth_labs", "regulated_by", "hipaa", "biosynth_privacy_policy.pdf", 300, 348, 0.93),

    ("azuregrid_energy", "produces", "gridbalance_controller", "azuregrid_datasheet.pdf", 70, 131, 0.95),
    ("azuregrid_energy", "uses", "solid_state_battery", "azuregrid_technology_review.pdf", 240, 301, 0.92),
    ("azuregrid_energy", "uses", "edge_inference", "azuregrid_edge_note.pdf", 112, 167, 0.90),
    ("azuregrid_energy", "collaborates_with", "mit_energy_initiative", "mit_azuregrid_project.pdf", 60, 121, 0.88),
    ("azuregrid_energy", "serves", "global_energy_storage", "azuregrid_investor_deck.pdf", 350, 402, 0.87),

    ("meridian_health", "produces", "meditrace_platform", "meridian_product_brief.pdf", 66, 125, 0.95),
    ("meridian_health", "uses", "federated_learning", "meridian_ml_whitepaper.pdf", 210, 271, 0.93),
    ("meridian_health", "uses", "edge_inference", "meridian_edge_deployment.pdf", 140, 192, 0.89),
    ("meridian_health", "regulated_by", "hipaa", "meridian_compliance_guide.pdf", 320, 369, 0.96),
    ("meridian_health", "serves", "us_healthcare_it", "meridian_market_map.pdf", 401, 448, 0.85),

    ("orion_semiconductors", "supplies", "edge_inference", "orion_supplier_catalog.pdf", 95, 148, 0.91),
    ("orion_semiconductors", "uses", "predictive_maintenance", "orion_fab_report.pdf", 180, 232, 0.88),
    ("orion_semiconductors", "regulated_by", "iso_27001", "orion_security_certificate.pdf", 260, 309, 0.94),

    ("vega_analytics", "uses", "federated_learning", "vega_platform_docs.pdf", 120, 178, 0.92),
    ("vega_analytics", "uses", "predictive_maintenance", "vega_industry_report.pdf", 230, 284, 0.90),
    ("vega_analytics", "serves", "eu_industrial_ai", "vega_go_to_market.pdf", 310, 357, 0.86),

    ("digital_twin", "enables", "fleetops_suite", "fleetops_architecture.pdf", 77, 129, 0.90),
    ("predictive_maintenance", "enables", "fleetops_suite", "fleetops_features.pdf", 150, 204, 0.89),
    ("federated_learning", "enables", "meditrace_platform", "meditrace_ml_design.pdf", 188, 243, 0.91),
    ("edge_inference", "enables", "gridbalance_controller", "gridbalance_edge_spec.pdf", 92, 144, 0.88),
    ("blockchain_settlement", "enables", "secureledger_api", "secureledger_tech_spec.pdf", 130, 186, 0.90),
    ("crispr_screening", "enables", "genemap_assay", "genemap_method.pdf", 201, 258, 0.93),
    ("synthetic_biology", "enables", "genemap_assay", "genemap_platform.pdf", 260, 315, 0.91),
    ("solid_state_battery", "enables", "gridbalance_controller", "gridbalance_battery_note.pdf", 110, 163, 0.89),

    ("eth_zurich_rl_lab", "researches", "federated_learning", "eth_rl_publication.pdf", 55, 112, 0.87),
    ("stanford_bio_x", "researches", "crispr_screening", "stanford_biotech_paper.pdf", 120, 179, 0.88),
    ("mit_energy_initiative", "researches", "solid_state_battery", "mit_battery_study.pdf", 180, 238, 0.89),
]

OBSERVED_COLUMNS = [
    "observed_id",
    "head",
    "relation",
    "tail",
    "source_document",
    "span_start",
    "span_end",
    "extractor_version",
    "confidence",
    "provenance_type",
    "ingested_at",
    "canonical_key",
]

CANDIDATE_COLUMNS = [
    "candidate_id",
    "head",
    "head_label",
    "head_type",
    "relation",
    "tail",
    "tail_label",
    "tail_type",
    "canonical_key",
    "common_neighbors",
    "adamic_adar",
    "jaccard",
    "description_overlap",
    "score",
    "confidence",
    "source_document",
    "span_start",
    "span_end",
    "extractor_version",
    "provenance_type",
    "ingested_at",
]


def iso_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def canonical_key(a: str, b: str) -> str:
    return "::".join(sorted((str(a), str(b))))


def stable_id(head: str, relation: str, tail: str) -> str:
    raw = f"{head}|{relation}|{tail}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:10]


def short_label(text: str, max_len: int = 16) -> str:
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


@st.cache_data(show_spinner=False)
def build_dataset(seed: int):
    entities = pd.DataFrame(ENTITY_RECORDS, columns=["id", "label", "type", "description"])
    entities["tokens"] = entities["description"].map(tokenize)

    valid_ids = set(entities["id"])
    obs_rows = []

    for idx, (head, relation, tail, source_doc, span_start, span_end, confidence) in enumerate(OBSERVED_TRIPLES):
        if head not in valid_ids or tail not in valid_ids:
            continue
        obs_rows.append(
            {
                "observed_id": f"obs_{idx:04d}",
                "head": head,
                "relation": relation,
                "tail": tail,
                "source_document": source_doc,
                "span_start": int(span_start),
                "span_end": int(span_end),
                "extractor_version": "demo-catalog/0.1.0",
                "confidence": float(confidence),
                "provenance_type": "observed",
                "ingested_at": iso_utc(BASE_TIME + idx * 43),
            }
        )

    observed = pd.DataFrame(obs_rows, columns=OBSERVED_COLUMNS)
    observed["canonical_key"] = observed.apply(lambda row: canonical_key(row["head"], row["tail"]), axis=1)
    observed_canon = set(observed["canonical_key"])

    graph = nx.Graph()
    graph.add_nodes_from(entities["id"])
    for row in observed.itertuples(index=False):
        graph.add_edge(row.head, row.tail, relation=row.relation)

    label_map = dict(zip(entities["id"], entities["label"]))
    type_map = dict(zip(entities["id"], entities["type"]))
    token_map = dict(zip(entities["id"], entities["tokens"]))

    candidate_rows = []
    skipped_existing = 0
    entity_records = entities.to_dict("records")

    for i, ei in enumerate(entity_records):
        for ej in entity_records[i + 1 :]:
            relation = None
            head_id = ei["id"]
            tail_id = ej["id"]

            if (ei["type"], ej["type"]) in RELATION_BY_TYPES:
                relation = RELATION_BY_TYPES[(ei["type"], ej["type"])]
            elif (ej["type"], ei["type"]) in RELATION_BY_TYPES:
                relation = RELATION_BY_TYPES[(ej["type"], ei["type"])]
                head_id = ej["id"]
                tail_id = ei["id"]
            else:
                continue

            canon = canonical_key(head_id, tail_id)

            if canon in observed_canon:
                skipped_existing += 1
                continue

            common_neighbor_list = list(nx.common_neighbors(graph, head_id, tail_id))
            cn = len(common_neighbor_list)

            deg_sum = graph.degree(head_id) + graph.degree(tail_id)
            union_deg = deg_sum - cn
            jaccard = cn / union_deg if union_deg > 0 else 0.0

            adamic_adar = 0.0
            for nb in common_neighbor_list:
                d = graph.degree(nb)
                if d > 1:
                    adamic_adar += 1.0 / math.log(d)

            tokens_a = token_map[head_id]
            tokens_b = token_map[tail_id]
            token_union = tokens_a | tokens_b
            description_overlap = len(tokens_a & tokens_b) / len(token_union) if token_union else 0.0

            score = (
                0.18
                + 0.30 * min(cn / 5.0, 1.0)
                + 0.18 * min(adamic_adar / 3.0, 1.0)
                + 0.08 * jaccard
                + 0.26 * description_overlap
            )
            score = max(0.0, min(1.0, score))
            confidence = max(0.05, min(0.99, 0.45 + 0.55 * score))
            cid = stable_id(head_id, relation, tail_id)

            candidate_rows.append(
                {
                    "candidate_id": cid,
                    "head": head_id,
                    "head_label": label_map[head_id],
                    "head_type": type_map[head_id],
                    "relation": relation,
                    "tail": tail_id,
                    "tail_label": label_map[tail_id],
                    "tail_type": type_map[tail_id],
                    "canonical_key": canon,
                    "common_neighbors": int(cn),
                    "adamic_adar": round(float(adamic_adar), 4),
                    "jaccard": round(float(jaccard), 4),
                    "description_overlap": round(float(description_overlap), 4),
                    "score": round(float(score), 4),
                    "confidence": round(float(confidence), 4),
                    "source_document": f"synthetic_demo://link_scorer/{cid}",
                    "span_start": 0,
                    "span_end": 64,
                    "extractor_version": "heuristic-link-scorer/0.1.0",
                    "provenance_type": "candidate",
                }
            )

    if candidate_rows:
        candidates = pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS)
        candidates = candidates.sort_values(["score", "candidate_id"], ascending=[False, True]).head(180).reset_index(drop=True)
    else:
        candidates = pd.DataFrame(columns=CANDIDATE_COLUMNS)

    candidates["ingested_at"] = [iso_utc(BASE_TIME + 90000 + i * 17) for i in range(len(candidates))]

    return entities, observed, candidates, skipped_existing


@st.cache_data(show_spinner=False)
def compute_simulation(threshold: float, max_new_per_step: int, steps: int, seed: int):
    entities, observed, candidates, skipped_existing = build_dataset(seed)

    active = set(observed["canonical_key"])
    product_owners = set(observed.loc[observed["relation"] == "produces", "tail"].tolist())

    discovered_rows = []
    current_step = 1
    added_in_step = 0

    if not candidates.empty:
        for row in candidates.itertuples(index=False):
            if current_step > steps:
                break

            if added_in_step >= max_new_per_step:
                current_step += 1
                added_in_step = 0
                continue

            if float(row.score) < threshold:
                continue

            if row.canonical_key in active:
                continue

            if row.relation == "produces" and row.tail in product_owners:
                continue

            active.add(row.canonical_key)

            if row.relation == "produces":
                product_owners.add(row.tail)

            record = row._asdict()
            record["step"] = int(current_step)
            record["review_status"] = "accepted_in_demo"
            discovered_rows.append(record)

            added_in_step += 1

    discovered_columns = CANDIDATE_COLUMNS + ["step", "review_status"]
    discovered = (
        pd.DataFrame(discovered_rows, columns=discovered_columns)
        if discovered_rows
        else pd.DataFrame(columns=discovered_columns)
    )

    if not candidates.empty:
        pending = candidates[~candidates["canonical_key"].isin(active)].head(25).copy()
    else:
        pending = candidates.copy()

    config_payload = {
        "threshold": round(float(threshold), 4),
        "max_new_per_step": int(max_new_per_step),
        "steps": int(steps),
        "seed": int(seed),
    }
    config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    avg_confidence = float(discovered["confidence"].mean()) if not discovered.empty else 0.0

    metrics = {
        "config_hash": config_hash,
        "seed": int(seed),
        "threshold": float(threshold),
        "max_new_per_step": int(max_new_per_step),
        "steps": int(steps),
        "entities": int(len(entities)),
        "observed_edges": int(len(observed)),
        "discovered_edges": int(len(discovered)),
        "candidate_pool": int(len(candidates)),
        "existing_pairs_blocked": int(skipped_existing),
        "average_discovered_confidence": float(avg_confidence),
        "pending_review": int(len(pending)),
        "generated_at": iso_utc(BASE_TIME + 120000 + (int(config_hash, 16) % 1000)),
    }

    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(entities["id"])

    for row in observed.itertuples(index=False):
        layout_graph.add_edge(row.head, row.tail)

    if not candidates.empty:
        for row in candidates.head(40).itertuples(index=False):
            layout_graph.add_edge(row.head, row.tail)

    pos = nx.spring_layout(layout_graph, seed=int(seed), k=1.7, iterations=120)
    pos_serial = {node: [float(xy[0]), float(xy[1])] for node, xy in pos.items()}

    return entities, observed, candidates, discovered, pending, metrics, pos_serial


def _try_write_mp4(path: Path, frames: list, fps: int) -> None:
    attempts = (
        {"fps": fps, "codec": "libx264", "pixelformat": "yuv420p", "macro_block_size": 1},
        {"fps": fps, "codec": "libx264", "macro_block_size": 1},
        {"fps": fps, "codec": "libx264"},
        {"fps": fps},
    )

    last_exc = None

    for kwargs in attempts:
        try:
            imageio.mimsave(path, frames, **kwargs)
            if path.exists() and path.stat().st_size > 0:
                return
            raise RuntimeError("MP4 writer produced an empty file.")
        except Exception as exc:
            last_exc = exc
            if path.exists():
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass

    raise RuntimeError(f"Unable to write MP4: {last_exc}")


def _try_write_gif(path: Path, frames: list, fps: int) -> None:
    attempts = (
        {"duration": 1.0 / float(fps), "loop": 0},
        {"fps": fps, "loop": 0},
        {"fps": fps},
    )

    last_exc = None

    for kwargs in attempts:
        try:
            imageio.mimsave(path, frames, **kwargs)
            if path.exists() and path.stat().st_size > 0:
                return
            raise RuntimeError("GIF writer produced an empty file.")
        except Exception as exc:
            last_exc = exc
            if path.exists():
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass

    raise RuntimeError(f"Unable to write GIF: {last_exc}")


def persist_frames(frames: list, stem: str, fps: int) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mp4_path = OUTPUT_DIR / f"{stem}.mp4"
    gif_path = OUTPUT_DIR / f"{stem}.gif"
    png_path = OUTPUT_DIR / f"{stem}.png"

    tmp_mp4 = OUTPUT_DIR / f".tmp_{stem}_{os.getpid()}.mp4"

    try:
        _try_write_mp4(tmp_mp4, frames, fps)
        os.replace(tmp_mp4, mp4_path)
        return {
            "path": str(mp4_path),
            "kind": "mp4",
            "message": "Generated MP4 demo video with imageio-ffmpeg.",
        }
    except Exception as exc:
        if tmp_mp4.exists():
            try:
                tmp_mp4.unlink(missing_ok=True)
            except Exception:
                pass

        tmp_gif = OUTPUT_DIR / f".tmp_{stem}_{os.getpid()}.gif"

        try:
            _try_write_gif(tmp_gif, frames, fps)
            os.replace(tmp_gif, gif_path)
            return {
                "path": str(gif_path),
                "kind": "gif",
                "message": f"MP4 encoding unavailable ({exc}); generated animated GIF fallback.",
            }
        except Exception as exc2:
            if tmp_gif.exists():
                try:
                    tmp_gif.unlink(missing_ok=True)
                except Exception:
                    pass

            try:
                Image.fromarray(frames[-1]).save(png_path)
                return {
                    "path": str(png_path),
                    "kind": "png",
                    "message": f"Video encoding failed ({exc2}); saved final frame as PNG fallback.",
                }
            except Exception as exc3:
                return {
                    "path": "",
                    "kind": "error",
                    "message": f"All artifact exports failed: {exc3}",
                }


def generate_video(sim: tuple, steps: int, show_candidates: bool, show_labels: bool, stem: str) -> dict:
    entities_df, observed_df, candidates_df, discovered_df, pending_df, metrics, pos = sim

    pos_np = {node: (float(xy[0]), float(xy[1])) for node, xy in pos.items()}
    label_map = dict(zip(entities_df["id"], entities_df["label"]))
    type_map = dict(zip(entities_df["id"], entities_df["type"]))

    observed_edges = list(zip(observed_df["head"], observed_df["tail"]))
    observed_keys = set(observed_df["canonical_key"])
    threshold = float(metrics["threshold"])

    base_graph = nx.Graph()
    base_graph.add_nodes_from(entities_df["id"])

    for head, tail in observed_edges:
        base_graph.add_edge(head, tail)

    if not candidates_df.empty:
        for row in candidates_df.itertuples(index=False):
            base_graph.add_edge(row.head, row.tail)

    if not discovered_df.empty:
        for row in discovered_df.itertuples(index=False):
            base_graph.add_edge(row.head, row.tail)

    def make_frame(step_idx: int):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax.set_axis_off()

        if observed_edges:
            nx.draw_networkx_edges(
                base_graph,
                pos=pos_np,
                ax=ax,
                edgelist=observed_edges,
                edge_color="#c7ccd6",
                width=1.3,
                alpha=0.9,
            )

        if step_idx > 0 and not discovered_df.empty:
            disc = discovered_df[discovered_df["step"] <= step_idx]
        else:
            disc = discovered_df.iloc[0:0]

        disc_edges = list(zip(disc["head"], disc["tail"])) if not disc.empty else []

        if disc_edges:
            nx.draw_networkx_edges(
                base_graph,
                pos=pos_np,
                ax=ax,
                edgelist=disc_edges,
                edge_color="#2ca02c",
                width=2.5,
                alpha=0.95,
            )

        candidate_frontier = []
        if show_candidates and not candidates_df.empty:
            active_keys = observed_keys | set(disc["canonical_key"])
            mask = (~candidates_df["canonical_key"].isin(active_keys)) & (candidates_df["score"] >= threshold)
            frontier_df = candidates_df[mask].head(6)
            candidate_frontier = list(zip(frontier_df["head"], frontier_df["tail"]))

            if candidate_frontier:
                nx.draw_networkx_edges(
                    base_graph,
                    pos=pos_np,
                    ax=ax,
                    edgelist=candidate_frontier,
                    edge_color="#ff7f0e",
                    style="dashed",
                    width=1.8,
                    alpha=0.8,
                )

        nodelist = list(pos_np.keys())
        node_colors = [TYPE_COLORS.get(type_map.get(node, ""), "#888888") for node in nodelist]

        nx.draw_networkx_nodes(
            base_graph,
            pos=pos_np,
            nodelist=nodelist,
            ax=ax,
            node_color=node_colors,
            node_size=430,
            edgecolors="white",
            linewidths=0.8,
        )

        if show_labels:
            labels = {node: short_label(label_map.get(node, node)) for node in nodelist}
            nx.draw_networkx_labels(
                base_graph,
                pos=pos_np,
                labels=labels,
                ax=ax,
                font_size=6,
                font_color="#222222",
            )

        legend_elements = [Patch(facecolor=color, label=etype) for etype, color in TYPE_COLORS.items()]
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.02),
            ncol=3,
            fontsize=7,
            frameon=False,
        )

        last_edge = "—"
        if not disc.empty:
            last = disc.iloc[-1]
            last_edge = (
                f"{short_label(label_map.get(last.head, last.head))} "
                f"--{last.relation}--> "
                f"{short_label(label_map.get(last.tail, last.tail))}"
            )

        ax.set_title(
            f"Automated Knowledge Discovery Engine — step {step_idx}/{steps}",
            fontsize=12,
            fontweight="bold",
        )

        info = (
            f"Threshold {threshold:.2f} | Discovered {len(disc)} | Frontier {len(candidate_frontier)}\n"
            f"Observed {len(observed_edges)} | Config {metrics['config_hash']}\n"
            f"Last: {last_edge}"
        )

        ax.text(
            0.01,
            0.01,
            info,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="#dddddd"),
        )

        fig.tight_layout()
        fig.canvas.draw()

        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

        plt.close(fig)
        return frame

    frames = [make_frame(i) for i in range(0, int(steps) + 1)]
    return persist_frames(frames, stem=stem, fps=6)


def get_video_info(sim: tuple, steps: int, show_candidates: bool, show_labels: bool) -> dict:
    metrics = sim[5]

    render_payload = {
        "sim_hash": metrics["config_hash"],
        "steps": int(steps),
        "show_candidates": bool(show_candidates),
        "show_labels": bool(show_labels),
    }
    render_hash = hashlib.sha1(json.dumps(render_payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    key = render_hash

    cache = st.session_state.setdefault("video_cache", {})

    if key in cache and cache[key].get("kind") == "error":
        return cache[key]

    if (
        key not in cache
        or not cache[key].get("path")
        or not Path(cache[key]["path"]).exists()
    ):
        with st.spinner("Generating live demo video..."):
            cache[key] = generate_video(
                sim=sim,
                steps=steps,
                show_candidates=show_candidates,
                show_labels=show_labels,
                stem=f"demo_link_discovery_{render_hash}",
            )

    return cache[key]


def render_video_widget(video_info: dict) -> None:
    path = video_info.get("path", "")
    kind = video_info.get("kind", "error")

    if kind == "error" or not path or not Path(path).exists():
        st.error(video_info.get("message", "Could not generate demo artifact."))
        return

    if kind == "mp4":
        try:
            st.video(path, loop=True, muted=True, autoplay=True)
        except TypeError:
            st.video(path, loop=True, muted=True)
    else:
        st.image(path, use_container_width=True)

    mime_map = {
        "mp4": "video/mp4",
        "gif": "image/gif",
        "png": "image/png",
    }

    st.download_button(
        label="Download generated demo file",
        data=Path(path).read_bytes(),
        file_name=Path(path).name,
        mime=mime_map.get(kind, "application/octet-stream"),
    )

    st.caption(f"Artifact: `{path}` — {video_info.get('message', '')}")


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "review_decisions" not in st.session_state:
        st.session_state.review_decisions = {}

    st.title(APP_TITLE)
    st.caption(
        "CPU-only, no API key, deterministic synthetic demo. The app auto-loads data, "
        "runs a link-discovery simulation, generates a playable video artifact in ./output, "
        "and exposes a human-review queue."
    )

    with st.sidebar:
        st.header("Simulation controls")

        threshold = st.slider(
            "Discovery threshold",
            min_value=0.20,
            max_value=0.90,
            value=0.34,
            step=0.01,
            help="Candidates below this score are not promoted in the demo simulation.",
        )

        max_new_per_step = st.slider(
            "Max new links per step",
            min_value=1,
            max_value=5,
            value=2,
            help="Limits how many candidate edges are accepted per simulation step.",
        )

        steps = st.slider(
            "Simulation steps",
            min_value=6,
            max_value=30,
            value=18,
            help="Number of animated discovery steps used for the generated video.",
        )

        show_candidates = st.checkbox("Show candidate frontier", value=True)
        show_labels = st.checkbox("Show node labels", value=True)

        seed = st.number_input(
            "Deterministic seed",
            min_value=1,
            max_value=999999999,
            value=SEED,
            step=1,
            format="%d",
        )

        st.divider()

        if st.button("Reset review decisions"):
            st.session_state.review_decisions = {}
            st.rerun()

        st.caption("Generated artifacts are written to `./output`.")

    sim = compute_simulation(
        threshold=float(threshold),
        max_new_per_step=int(max_new_per_step),
        steps=int(steps),
        seed=int(seed),
    )

    entities_df, observed_df, candidates_df, discovered_df, pending_df, metrics, pos = sim
    video_info = get_video_info(sim, int(steps), bool(show_candidates), bool(show_labels))

    tab_demo, tab_review, tab_provenance, tab_checks = st.tabs(
        ["🎬 Live demo", "🧾 Review queue", "🔍 Provenance", "🛡️ Leakage and robustness"]
    )

    with tab_demo:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Entities", metrics["entities"])
        col2.metric("Observed edges", metrics["observed_edges"])
        col3.metric("Discovered edges", metrics["discovered_edges"])
        col4.metric("Avg confidence", f"{metrics['average_discovered_confidence']:.2f}")
        col5.metric("Config hash", metrics["config_hash"])

        st.subheader("Auto-generated working demo")
        render_video_widget(video_info)

        st.subheader("Discovered edges in this simulation run")
        if discovered_df.empty:
            st.info("No edges crossed the current discovery threshold. Lower the threshold or increase steps.")
        else:
            display_cols = [
                "step",
                "head_label",
                "relation",
                "tail_label",
                "score",
                "confidence",
                "common_neighbors",
                "description_overlap",
                "review_status",
            ]
            st.dataframe(discovered_df[display_cols].head(100), use_container_width=True)

    with tab_review:
        st.subheader("Human review queue")
        st.write(
            "Candidate links are treated as proposals, not facts. Approve or reject items below. "
            "Decisions are stored locally for this session and can be downloaded as JSON."
        )

        if pending_df.empty:
            st.success("No pending candidates remain in the top review slice.")
        else:
            for _, row in pending_df.head(20).iterrows():
                cid = row["candidate_id"]
                status = st.session_state.review_decisions.get(cid, "pending")

                c1, c2, c3, c4 = st.columns([6, 1, 1, 1])

                c1.markdown(
                    f"**{row['head_label']}** `{row['relation']}` **{row['tail_label']}** — "
                    f"score `{row['score']:.2f}`, confidence `{row['confidence']:.2f}`, "
                    f"common neighbors `{row['common_neighbors']}`"
                )

                c2.metric("Status", status)

                if c3.button("Approve", key=f"approve_{cid}"):
                    st.session_state.review_decisions[cid] = "approved"
                    st.rerun()

                if c4.button("Reject", key=f"reject_{cid}"):
                    st.session_state.review_decisions[cid] = "rejected"
                    st.rerun()

        st.divider()

        decision_payload = {
            "generated_at": iso_utc(time.time()),
            "config_hash": metrics["config_hash"],
            "threshold": metrics["threshold"],
            "decisions": st.session_state.review_decisions,
        }

        st.download_button(
            label="Download review decisions",
            data=json.dumps(decision_payload, indent=2),
            file_name="review_decisions.json",
            mime="application/json",
        )

    with tab_provenance:
        st.subheader("Observed edge provenance")
        observed_display = [
            "observed_id",
            "head",
            "relation",
            "tail",
            "source_document",
            "span_start",
            "span_end",
            "extractor_version",
            "confidence",
            "ingested_at",
        ]
        st.dataframe(observed_df[observed_display].head(100), use_container_width=True)

        st.subheader("Candidate edge provenance")
        candidate_display = [
            "candidate_id",
            "head_label",
            "relation",
            "tail_label",
            "score",
            "confidence",
            "source_document",
            "extractor_version",
            "provenance_type",
            "ingested_at",
        ]
        st.dataframe(candidates_df[candidate_display].head(100), use_container_width=True)

        st.subheader("Discovered edge provenance")
        if discovered_df.empty:
            st.info("No discovered edges in the current run.")
        else:
            discovered_display = [
                "candidate_id",
                "head_label",
                "relation",
                "tail_label",
                "score",
                "confidence",
                "step",
                "review_status",
                "source_document",
                "extractor_version",
                "ingested_at",
            ]
            st.dataframe(discovered_df[discovered_display].head(100), use_container_width=True)

    with tab_checks:
        st.subheader("Leakage and safety checks")

        st.markdown(
            """
            - Candidate generation excludes canonical pairs already present in the observed graph.
            - Inverse and duplicate triples are collapsed to a canonical sorted pair before checking.
            - The demo uses synthetic data only; no private documents or external API calls are used.
            - Generated artifacts are written atomically to temporary files and then replaced.
            - Heuristic candidate outputs are treated as proposals, not facts.
            - Human review decisions are kept separate from the simulated accepted graph.
            """
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Existing pairs blocked", metrics["existing_pairs_blocked"])
        col2.metric("Candidate pool", metrics["candidate_pool"])
        col3.metric("Pending review", metrics["pending_review"])
        col4.metric("Seed", metrics["seed"])

        st.subheader("Examples of observed pairs blocked from candidate generation")
        if observed_df.empty:
            st.info("No observed edges available.")
        else:
            st.dataframe(
                observed_df[["head", "relation", "tail", "canonical_key", "source_document"]].head(20),
                use_container_width=True,
            )

        st.subheader("Runtime and artifact metadata")
        st.json(
            {
                "config_hash": metrics["config_hash"],
                "threshold": metrics["threshold"],
                "max_new_per_step": metrics["max_new_per_step"],
                "steps": metrics["steps"],
                "generated_at": metrics["generated_at"],
                "artifact_path": video_info.get("path", ""),
                "artifact_kind": video_info.get("kind", ""),
                "artifact_message": video_info.get("message", ""),
            }
        )


if __name__ == "__main__":
    main()