const express = require('express');
const router = express.Router();

// Mock prediction for now - will be replaced with ML model
const mockPrediction = (customerData) => {
  // Simple mock logic based on survey responses
  let score = 0.5; // Base score
  
  // Adjust based on factors from your Google Form
  if (customerData.nps_score <= 5) score += 0.2;
  if (customerData.content_satisfaction <= 2) score += 0.15;
  if (customerData.value_perception === 'Poor value') score += 0.15;
  if (customerData.usage_frequency === 'Rarely (once in 2-3 months)') score += 0.1;
  if (customerData.churn_likelihood === 'Very likely') score += 0.2;
  if (customerData.switching_behavior.includes('Yes')) score += 0.1;
  
  // Cap at 0.95
  score = Math.min(0.95, Math.max(0.05, score));
  
  return {
    churn_probability: score,
    risk_level: score > 0.7 ? 'High' : score > 0.4 ? 'Medium' : 'Low',
    key_factors: [
      customerData.nps_score <= 5 ? 'Low recommendation score' : null,
      customerData.content_satisfaction <= 2 ? 'Content dissatisfaction' : null,
      customerData.value_perception === 'Poor value' ? 'Poor value perception' : null,
      customerData.usage_frequency === 'Rarely (once in 2-3 months)' ? 'Infrequent usage' : null
    ].filter(factor => factor !== null),
    recommendations: [
      score > 0.7 ? 'Offer immediate discount or special offer' : null,
      score > 0.5 ? 'Send personalized content recommendations' : null,
      score > 0.3 ? 'Conduct satisfaction survey' : null,
      'Consider loyalty program enrollment'
    ].filter(rec => rec !== null),
    confidence: 0.85 // Mock confidence score
  };
};

// @route   POST /api/predict/churn
// @desc    Predict churn probability for customer data
// @access  Public
router.post('/churn', (req, res) => {
  try {
    const customerData = req.body;
    
    // Basic validation
    if (!customerData || Object.keys(customerData).length === 0) {
      return res.status(400).json({
        success: false,
        message: 'Customer data is required'
      });
    }
    
    const prediction = mockPrediction(customerData);
    
    res.json({
      success: true,
      message: 'Prediction generated successfully',
      prediction,
      customer_data: customerData,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({
      success: false,
      message: 'Error generating prediction',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// @route   GET /api/predict/mock
// @desc    Get mock prediction for testing
// @access  Public
router.get('/mock', (req, res) => {
  const mockData = {
    age_group: '18-35',
    nps_score: 3,
    content_satisfaction: 2,
    value_perception: 'Poor value',
    usage_frequency: 'Rarely (once in 2-3 months)',
    churn_likelihood: 'Very likely',
    switching_behavior: 'Yes, switched completely'
  };
  
  const prediction = mockPrediction(mockData);
  
  res.json({
    success: true,
    message: 'Mock prediction for testing',
    sample_data: mockData,
    prediction
  });
});

module.exports = router;