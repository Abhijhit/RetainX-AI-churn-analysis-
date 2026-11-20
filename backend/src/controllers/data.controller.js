const { Customer, Subscription } = require('../models');
const { mockCustomers } = require('../utils/mockData');

// Submit customer data (form)
const submitCustomerData = async (req, res) => {
  try {
    const customerData = req.body;
    console.log('📝 Received data:', customerData);
    
    // Validate required fields
    const requiredFields = ['age_group', 'subscription_duration', 'nps_score', 'churn_likelihood'];
    for (const field of requiredFields) {
      if (!customerData[field]) {
        console.log(`❌ Missing field: ${field}`);
        return res.status(400).json({
          success: false,
          message: `Missing required field: ${field}`
        });
      }
    }
    
    // Add mock prediction for now (will be replaced with ML model)
    customerData.predicted_churn_probability = Math.random();
    customerData.churn_risk_category = customerData.predicted_churn_probability > 0.7 ? 'High' : 
                                      customerData.predicted_churn_probability > 0.4 ? 'Medium' : 'Low';
    
    console.log('💾 Attempting to save to database...');
    
    // Save to database
    const customer = await Customer.create(customerData);
    
    console.log('✅ Customer saved successfully, ID:', customer.id);
    
    res.status(201).json({
      success: true,
      message: 'Customer data saved successfully',
      data: customer,
      prediction: {
        churn_probability: customerData.predicted_churn_probability,
        risk_category: customerData.churn_risk_category
      }
    });
    
  } catch (error) {
    console.error('❌ Error saving customer data:', error.message);
    console.error('📋 Full error:', error);
    res.status(500).json({
      success: false,
      message: 'Error saving customer data: ' + error.message,
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

// Get all customers
const getAllCustomers = async (req, res) => {
  try {
    const customers = await Customer.findAll({
      order: [['created_at', 'DESC']],
      limit: 100
    });
    
    res.json({
      success: true,
      count: customers.length,
      data: customers
    });
    
  } catch (error) {
    console.error('Error fetching customers:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching customers'
    });
  }
};

// Get customer by ID
const getCustomerById = async (req, res) => {
  try {
    const customer = await Customer.findByPk(req.params.id);
    
    if (!customer) {
      return res.status(404).json({
        success: false,
        message: 'Customer not found'
      });
    }
    
    res.json({
      success: true,
      data: customer
    });
    
  } catch (error) {
    console.error('Error fetching customer:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching customer'
    });
  }
};

// Get mock customers (for development)
const getMockCustomers = (req, res) => {
  res.json({
    success: true,
    message: 'Mock data for development',
    count: mockCustomers.length,
    data: mockCustomers
  });
};

module.exports = {
  submitCustomerData,
  getAllCustomers,
  getCustomerById,
  getMockCustomers
};