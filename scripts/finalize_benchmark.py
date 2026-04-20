import json

# Explicitly selected High-Signal IDs
GOLDEN_IDS = {
    "numeric": [
        "num_AAPL_NetIncomeLoss_2025-12-27",
        "num_AAPL_NetCashProvidedByUsedInOperatingActivities_2025-12-27",
        "num_MSFT_NetIncomeLoss_2025-12-31",
        "num_MSFT_GrossProfit_2025-12-31",
        "num_MSFT_NetCashProvidedByUsedInOperatingActivities_2025-12-31",
        "num_GOOGL_Revenues_2025-12-31",
        "num_GOOGL_PaymentsToAcquirePropertyPlantAndEquipment_2025-12-31",
        "num_NVDA_Revenues_2026-01-25",
        "num_NVDA_NetIncomeLoss_2026-01-25",
        "num_NVDA_GrossProfit_2026-01-25",
        "num_TSLA_Revenues_2025-12-31",
        "num_TSLA_NetIncomeLoss_2025-12-31",
        "num_AMZN_NetIncomeLoss_2025-12-31",
        "num_META_OperatingIncomeLoss_2025-12-31",
        "num_INTC_NetIncomeLoss_2025-12-27"
    ],
    "risk": [
        "risk_AAPL_competition",
        "risk_AAPL_regulation",
        "risk_AAPL_supply_chain",
        "risk_MSFT_competition",
        "risk_MSFT_regulation",
        "risk_GOOGL_regulation",
        "risk_NVDA_supply_chain",
        "risk_NVDA_competition",
        "risk_NVDA_regulation",
        "risk_TSLA_supply_chain",
        "risk_TSLA_competition",
        "risk_AMZN_competition",
        "risk_META_regulation",
        "risk_AMD_supply_chain",
        "risk_INTC_supply_chain"
    ],
    "news_sentiment": [
        "news_AAPL_3",
        "news_MSFT_3",
        "news_GOOGL_4",
        "news_NVDA_0",
        "news_NVDA_1",
        "news_TSLA_1",
        "news_TSLA_2",
        "news_AMZN_3",
        "news_META_0",
        "news_INTC_1"
    ],
    "earnings_sentiment": [
        "earn_AAPL_Q1_2025",
        "earn_AAPL_Q4_2024",
        "earn_TSLA_Q1_2025",
        "earn_MSFT_placeholder",
        "earn_GOOGL_placeholder",
        "earn_NVDA_placeholder",
        "earn_AMZN_placeholder",
        "earn_META_placeholder",
        "earn_NFLX_placeholder",
        "earn_AMD_placeholder"
    ]
}

def finalize_benchmark(input_path, output_path):
    with open(input_path, "r") as f:
        candidates = [json.loads(line) for line in f]
    
    # Map for easy lookup
    candidate_map = {c['id']: c for c in candidates}
    
    final_benchmark = []
    
    # Collect IDs in order of distribution
    for category, ids in GOLDEN_IDS.items():
        for cid in ids:
            if cid in candidate_map:
                final_benchmark.append(candidate_map[cid])
            else:
                print(f"Warning: ID {cid} not found in candidates. Check generation script.")

    print(f"Final Golden Benchmark size: {len(final_benchmark)}")
    
    with open(output_path, "w") as f:
        for c in final_benchmark:
            f.write(json.dumps(c) + "\n")

if __name__ == "__main__":
    finalize_benchmark("evaluation/candidates.jsonl", "evaluation/benchmark_v1.jsonl")
