const express = require('express');
const router = express.Router();
const { login, register, getProfile } = require('../controllers/auth.controller');

// @route   POST /api/auth/login
// @desc    Authenticate user
// @access  Public
router.post('/login', login);

// @route   POST /api/auth/register
// @desc    Register new user
// @access  Public
router.post('/register', register);

// @route   GET /api/auth/profile
// @desc    Get user profile
// @access  Private (mock for now)
router.get('/profile', getProfile);

module.exports = router;