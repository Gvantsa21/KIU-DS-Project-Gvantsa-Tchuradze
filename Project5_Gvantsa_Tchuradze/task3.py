"""
Task 3: Business Insights & Recommendations

Part A: Executive Summary

Problem Statement
The company is experiencing significant customer churn, where many customers
stop purchasing over time. Customer churn reduces revenue, increases customer
acquisition costs, and weakens long-term customer lifetime value. Predicting
customers at risk of churn allows the company to take proactive and cost-
effective retention actions.

Approach
A supervised classification approach was applied to predict customer churn.
Several machine-learning models were implemented, including Logistic Regression,
Decision Tree, Random Forest, and Support Vector Machine. The analysis was based
on data from 1,000 e-commerce customers, incorporating demographics, purchasing
behavior, engagement metrics, satisfaction scores, and loyalty indicators.

Key Findings
- The Random Forest model achieved the best performance, with the highest ROC-AUC
  and balanced precision–recall.
- Customers with long inactivity periods, low purchase frequency, and low
  satisfaction scores are significantly more likely to churn.
- Basic membership customers show much higher churn rates compared to Premium
  and VIP members.
- Engagement metrics such as email open rate and website visits are strongly
  associated with customer retention.
- More than 50% of customers are classified as churned or high-risk, indicating
  substantial revenue exposure.

Bottom Line
Customer churn is both predictable and actionable. Targeted retention strategies
focused on inactive and low-engagement customers, combined with stronger loyalty
programs, can significantly reduce revenue loss and increase customer lifetime
value.




Part B: Detailed Analysis (Classification)

1. Key Churn Drivers
The strongest predictors of churn include:
- Days Since Last Purchase (most influential factor)
- Purchase Frequency
- Customer Satisfaction Score and Net Promoter Score (NPS)
- Engagement metrics such as email opens and website visits
- Membership Type, with Basic members being more likely to churn

Typical churned customer profile:
Low engagement, infrequent purchases, long inactivity periods, lower satisfaction,
and Basic membership status.

2. Model Performance
Best-performing model: Random Forest
- High ROC-AUC indicates strong ability to distinguish between churned and active
  customers.
- Precision–recall balance is suitable for business decision-making.
- Predictions are reliable for targeted retention campaigns rather than fully
  automated decisions.

3. At-Risk Customers & Revenue Exposure
- Approximately 520 customers are identified as churned or high-risk.
- Average customer spending is approximately $2,000.
- Estimated revenue at risk:
  520 × $2,000 ≈ $1,040,000.


  
Part C: Actionable Recommendations

1. Immediate Actions (Next 30 Days)
- Launch retention campaigns for customers inactive for more than 90 days.
- Offer targeted discounts to low-frequency buyers.
- Initiate support outreach for customers with low satisfaction or frequent
  complaints.
- Prioritize Basic members for engagement incentives.

2. Strategic Initiatives (Next 6 Months)
- Strengthen loyalty benefits for Premium and VIP membership tiers.
- Integrate churn-risk scoring into the CRM system.
- Improve post-purchase communication and engagement workflows.

3. Personalized Retention Strategy
- High-risk customers: Direct outreach with strong incentives.
- Medium-risk customers: Engagement emails and loyalty point rewards.
- Low-risk customers: Upselling opportunities and referral programs.

4. ROI Estimation
If 20% of at-risk customers are retained:
104 × $2,000 ≈ $208,000 in preserved revenue.

5. Implementation Plan
- Owners: Marketing (campaign execution), Data Team (model maintenance),
  Customer Support (customer outreach).
- Resources: CRM integration, email automation tools, analytics tracking.
- Success Metrics: Reduced churn rate, increased repeat purchases, and improved
  customer satisfaction scores.
"""
