"""Lab 14a: compare monthly cost and $/1M tokens for EKS, SageMaker and a token-priced API.

    python3 cost_compare.py --tokens-per-day 200e6 --peak-to-avg 3 \
        --replica-tok-s 2500 --eks-instance-hour 1.0 --sm-instance-hour 1.3 --api-per-mtok 0.5

All prices are inputs: look them up for your region and model (they change).
The model makes the key structural differences explicit:
  - self-hosted capacity is provisioned for PEAK traffic (minus what autoscaling recovers)
  - token-priced APIs bill only what you use, but at a per-token margin
  - platform cost is not just instances: people, control plane, monitoring, idle headroom
"""

import argparse
import math


def self_hosted(tokens_per_day, peak_to_avg, replica_tok_s, instance_hour, autoscale_efficiency,
                min_replicas, fixed_monthly, util_target):
    avg_tok_s = tokens_per_day / 86400
    peak_tok_s = avg_tok_s * peak_to_avg
    peak_replicas = max(min_replicas, math.ceil(peak_tok_s / (replica_tok_s * util_target)))
    avg_replicas_needed = max(min_replicas, avg_tok_s / (replica_tok_s * util_target))
    # autoscale_efficiency: 0 = always provisioned for peak, 1 = perfectly follows demand
    replicas = peak_replicas - autoscale_efficiency * (peak_replicas - avg_replicas_needed)
    monthly = replicas * instance_hour * 730 + fixed_monthly
    return monthly, peak_replicas, replicas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens-per-day", type=float, default=200e6, help="output tokens per day")
    p.add_argument("--peak-to-avg", type=float, default=3.0)
    p.add_argument("--replica-tok-s", type=float, default=2500, help="measured goodput per replica AT your SLO (level 9)")
    p.add_argument("--util-target", type=float, default=0.7, help="plan to run replicas at this fraction of goodput")
    p.add_argument("--eks-instance-hour", type=float, default=1.0)
    p.add_argument("--sm-instance-hour", type=float, default=1.3, help="SageMaker ml.* price for the same hardware")
    p.add_argument("--eks-fixed-monthly", type=float, default=600, help="control plane, system nodes, monitoring, LB")
    p.add_argument("--sm-fixed-monthly", type=float, default=100)
    p.add_argument("--eks-autoscale-eff", type=float, default=0.6)
    p.add_argument("--sm-autoscale-eff", type=float, default=0.5)
    p.add_argument("--min-replicas", type=int, default=2, help="for availability across AZs")
    p.add_argument("--api-per-mtok", type=float, default=0.5, help="blended $/1M tokens of a hosted API")
    a = p.parse_args()

    monthly_tokens = a.tokens_per_day * 30.4
    rows = []
    for name, hour, fixed, eff in [("EKS + vLLM", a.eks_instance_hour, a.eks_fixed_monthly, a.eks_autoscale_eff),
                                   ("SageMaker + vLLM", a.sm_instance_hour, a.sm_fixed_monthly, a.sm_autoscale_eff)]:
        m, peak, avg = self_hosted(a.tokens_per_day, a.peak_to_avg, a.replica_tok_s, hour, eff,
                                   a.min_replicas, fixed, a.util_target)
        rows.append((name, m, f"peak {peak} replicas, avg {avg:.1f}"))
    rows.append(("Token-priced API", monthly_tokens / 1e6 * a.api_per_mtok, "no capacity to manage"))

    print(f"{monthly_tokens / 1e9:.1f}B output tokens/month, peak/avg {a.peak_to_avg}x\n")
    print(f"{'option':<20}{'$/month':>12}{'$/1M tok':>10}  notes")
    for name, m, note in rows:
        print(f"{name:<20}{m:>12,.0f}{m / (monthly_tokens / 1e6):>10.3f}  {note}")

    # break-even volume for EKS vs API
    lo, hi = 1e5, 1e12
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        m, _, _ = self_hosted(mid, a.peak_to_avg, a.replica_tok_s, a.eks_instance_hour, a.eks_autoscale_eff,
                              a.min_replicas, a.eks_fixed_monthly, a.util_target)
        api = mid * 30.4 / 1e6 * a.api_per_mtok
        lo, hi = (mid, hi) if m > api else (lo, mid)
    print(f"\nEKS becomes cheaper than the API above ~{lo / 1e6:,.0f}M output tokens/day (with these inputs)."
          "\nMissing from all rows: engineer time. One on-call rotation can outweigh the GPU bill at small scale.")


if __name__ == "__main__":
    main()
