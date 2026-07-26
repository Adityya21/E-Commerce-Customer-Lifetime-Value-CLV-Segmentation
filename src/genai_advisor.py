"""
GenAI Retention Advisor Module
================================
Uses an LLM (via Groq API) to generate personalized
retention strategies and marketing copy per customer segment.

Differentiators from Bank-Churn-AI-Advisor:
1. Segment-level (not individual) — scalable business advice
2. Revenue-impact estimates tied to CLV predictions
3. Actual marketing copy generation (email subjects, hooks)
4. Persona-specific tone matching
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


SYSTEM_PROMPT = """You are a senior retention strategist for a premium multi-category e-commerce brand
that sells Electronics, Fashion, Beauty, Home Decor, Sports gear, Footwear, and Accessories
across 7 countries (USA, UK, Canada, India, UAE, Germany, Australia).

You have deep expertise in customer lifetime value optimization, churn prevention,
and data-driven marketing strategy.

When given a customer segment profile (with metrics like avg CLV, recency, frequency,
return rate, top categories, and SHAP-based CLV drivers), you must generate:

1. **Segment Insight** (2-3 sentences): What defines this group psychologically and behaviorally.

2. **Three Retention Tactics** — each with:
   - Tactic name
   - Description (1-2 sentences)
   - Estimated revenue impact (e.g., "Could reduce churn by 15-20%, preserving ~$X,XXX in annual revenue per 100 customers")
   - Implementation difficulty (Low / Medium / High)

3. **Email Campaign Hook**:
   - Subject line (max 50 chars, high-open-rate style)
   - Body hook (2 sentences max, persona-appropriate tone)

4. **Risk Assessment**: Rate this segment as Low / Medium / High / Critical risk,
   with a 1-sentence justification tied to their metrics.

CRITICAL TONE RULES:
- Premium Loyalists → exclusive, VIP, aspirational language
- At-Risk Whales → urgency + win-back, acknowledge their past value
- Deal Seekers → value-driven, scarcity, "smart shopper" framing
- Dormant Browsers → curiosity hooks, "we miss you", fresh starts
- Rising Champions → encouragement, early access, growth recognition

Format your response as valid JSON with this structure:
{
  "segment_insight": "...",
  "retention_tactics": [
    {"name": "...", "description": "...", "revenue_impact": "...", "difficulty": "..."},
    ...
  ],
  "email_campaign": {
    "subject_line": "...",
    "body_hook": "..."
  },
  "risk_assessment": {
    "level": "...",
    "justification": "..."
  }
}
"""


def build_segment_prompt(profile, shap_drivers=None):
    """
    Build a user-prompt with segment data for the LLM.

    Parameters
    ----------
    profile : dict
        Segment profile from clustering module.
    shap_drivers : list, optional
        Top SHAP features driving CLV for this segment.

    Returns
    -------
    str: formatted prompt
    """
    prompt = f"""
Analyze this customer segment and generate retention strategies:

**Segment: {profile['persona_name']}**
- Size: {profile['size']:,} customers ({profile['pct_of_base']}% of total base)
- Avg CLV: ${profile['avg_clv']:,.2f}
- Total Segment Revenue: ${profile['total_revenue']:,.2f}
- Avg Recency: {profile['avg_recency']:.0f} days since last purchase
- Avg Frequency: {profile['avg_frequency']:.1f} orders per customer
- Avg Monetary (per order): ${profile['avg_monetary']:,.2f}
- Avg Tenure: {profile['avg_tenure_days']:.0f} days
- Return Rate: {profile['avg_return_rate']:.1f}%
- Avg Discount Used: {profile['avg_discount']:.0f}%
- Category Diversity: {profile['avg_category_diversity']:.1f} categories
- Avg Rating Given: {profile['avg_rating']:.1f}/5.0
- Top Category: {profile['top_category']}
- Top Acquisition Channel: {profile['top_channel']}
"""

    if shap_drivers:
        prompt += "\n**Top CLV Drivers (SHAP Analysis):**\n"
        for driver in shap_drivers[:5]:
            direction = "↑" if driver["avg_shap_value"] > 0 else "↓"
            prompt += f"  {direction} {driver['feature']}: {driver['avg_shap_value']:.4f} impact\n"

    prompt += "\nGenerate your analysis as the specified JSON format."
    return prompt


def query_llm(system_prompt, user_prompt, provider=None):
    """
    Send prompt to LLM via Groq API.

    Returns
    -------
    dict: parsed JSON response from LLM
    """
    provider = provider or config.GENAI_PROVIDER

    if provider == "groq":
        return _query_groq(system_prompt, user_prompt)
    else:
        raise ValueError(f"Unknown GenAI provider: {provider}")


def _query_groq(system_prompt, user_prompt):
    """Query Groq API."""
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )

    text = response.choices[0].message.content
    return json.loads(text)


def generate_retention_strategy(profile, shap_drivers=None, provider=None):
    """
    High-level function: generate retention strategy for a segment.

    Parameters
    ----------
    profile : dict
        Segment profile from clustering.
    shap_drivers : list, optional
        SHAP feature importance for this segment.
    provider : str, optional
        "groq". Defaults to config.

    Returns
    -------
    dict: LLM-generated retention strategy
    """
    user_prompt = build_segment_prompt(profile, shap_drivers)

    try:
        result = query_llm(SYSTEM_PROMPT, user_prompt, provider)
        result["_status"] = "success"
        result["_segment"] = profile["persona_name"]
        return result
    except Exception as e:
        # Graceful fallback — don't crash the app if LLM is unavailable
        return {
            "_status": "error",
            "_error": str(e),
            "_segment": profile["persona_name"],
            "segment_insight": f"Unable to generate AI insight for {profile['persona_name']}. Please check your API key configuration.",
            "retention_tactics": [],
            "email_campaign": {"subject_line": "N/A", "body_hook": "N/A"},
            "risk_assessment": {"level": "Unknown", "justification": "LLM unavailable"},
        }
