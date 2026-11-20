const { Sequelize } = require('sequelize');
require('dotenv').config();
const path = require('path');

// Use SQLite for development
const sequelize = new Sequelize({
  dialect: 'sqlite',
  storage: path.join(__dirname, '../../database.sqlite'),
  logging: process.env.NODE_ENV === 'development' ? console.log : false,
  define: {
    timestamps: true,
    createdAt: 'created_at',
    updatedAt: 'updated_at'
  }
});

// Test database connection
const testConnection = async () => {
  try {
    await sequelize.authenticate();
    console.log('✅ SQLite database connection established successfully.');
    console.log(`📁 Database file: ${path.join(__dirname, '../../database.sqlite')}`);
    return true;
  } catch (error) {
    console.error('❌ Unable to connect to SQLite database:', error.message);
    return false;
  }
};

module.exports = {
  sequelize,
  testConnection
};