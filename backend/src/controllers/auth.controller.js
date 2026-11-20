// Mock authentication for now - will implement JWT later
const login = (req, res) => {
    const { email, password } = req.body;
    
    // Mock validation
    if (!email || !password) {
      return res.status(400).json({
        success: false,
        message: 'Email and password are required'
      });
    }
    
    // Mock user
    const mockUser = {
      id: 1,
      email: 'admin@churnanalysis.com',
      name: 'Admin User',
      role: 'admin',
      token: 'mock-jwt-token-for-now'
    };
    
    res.json({
      success: true,
      message: 'Login successful',
      user: mockUser
    });
  };
  
  const register = (req, res) => {
    const { name, email, password } = req.body;
    
    if (!name || !email || !password) {
      return res.status(400).json({
        success: false,
        message: 'Name, email and password are required'
      });
    }
    
    // Mock registration
    res.status(201).json({
      success: true,
      message: 'Registration successful',
      user: {
        id: Date.now(),
        name,
        email,
        role: 'user'
      }
    });
  };
  
  const getProfile = (req, res) => {
    // Mock profile
    res.json({
      success: true,
      user: {
        id: 1,
        name: 'Admin User',
        email: 'admin@churnanalysis.com',
        role: 'admin',
        joined: '2024-01-01'
      }
    });
  };
  
  module.exports = {
    login,
    register,
    getProfile
  };