// Mock data for development and testing
const mockCustomers = [
    {
      age_group: '18-35',
      subscription_duration: '1-2 years',
      subscription_plan: 'Premium',
      monthly_spend: '₹200 - ₹500',
      usage_frequency: 'Daily',
      nps_score: 8,
      content_satisfaction: 4,
      value_perception: 'Excellent value',
      churn_likelihood: 'Very unlikely',
      switching_behavior: 'No changes',
      switching_reason: '',
      account_sharing: 'Yes, with family',
      competitor_awareness: 'No, satisfied with current',
      bundling_preference: 'Yes, definitely',
      payment_method: 'Monthly auto-renewal',
      suggestions: 'Great content library',
      predicted_churn_probability: 0.15,
      churn_risk_category: 'Low'
    },
    {
      age_group: '18-35',
      subscription_duration: '3-6 months',
      subscription_plan: 'Basic',
      monthly_spend: 'Less than ₹200',
      usage_frequency: 'Rarely (once in 2-3 months)',
      nps_score: 3,
      content_satisfaction: 2,
      value_perception: 'Poor value',
      churn_likelihood: 'Very likely',
      switching_behavior: 'Yes, switched completely',
      switching_reason: 'Too expensive / Price hike',
      account_sharing: 'No, only I use it',
      competitor_awareness: 'Yes, and actively considering switch',
      bundling_preference: 'No, prefer separate',
      payment_method: 'Pay as you go (monthly manual)',
      suggestions: 'Too expensive for limited content',
      predicted_churn_probability: 0.85,
      churn_risk_category: 'High'
    }
  ];
  
  const mockPredictions = [
    {
      customer_id: 1,
      churn_probability: 0.15,
      risk_level: 'Low',
      key_factors: ['High satisfaction', 'Frequent usage', 'Good value perception'],
      recommendations: ['Continue current strategy', 'Offer loyalty bonus']
    },
    {
      customer_id: 2,
      churn_probability: 0.85,
      risk_level: 'High',
      key_factors: ['Low satisfaction', 'Infrequent usage', 'Price sensitivity'],
      recommendations: ['Offer discount', 'Personalized content recommendations', 'Survey for feedback']
    }
  ];
  
  module.exports = {
    mockCustomers,
    mockPredictions
  };