const express = require('express');
const router = express.Router();
const { 
  submitCustomerData, 
  getAllCustomers, 
  getCustomerById,
  getMockCustomers 
} = require('../controllers/data.controller');

// @route   POST /api/data/customer
// @desc    Submit customer data (form submission)
// @access  Public
router.post('/customer', submitCustomerData);

// @route   GET /api/data/customers
// @desc    Get all customers
// @access  Public
router.get('/customers', getAllCustomers);

// @route   GET /api/data/customers/mock
// @desc    Get mock customers for development
// @access  Public
router.get('/customers/mock', getMockCustomers);

// @route   GET /api/data/customers/:id
// @desc    Get customer by ID
// @access  Public
router.get('/customers/:id', getCustomerById);

module.exports = router;