const { sequelize } = require('../config/database');
const { DataTypes } = require('sequelize');

// Customer Model (main table)
const Customer = sequelize.define('Customer', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  age_group: {
    type: DataTypes.STRING(10),
    allowNull: false
  },
  subscription_duration: {
    type: DataTypes.STRING(20),
    allowNull: false
  },
  subscription_plan: {
    type: DataTypes.STRING(20),
    allowNull: false
  },
  monthly_spend: {
    type: DataTypes.STRING(20),
    allowNull: false
  },
  usage_frequency: {
    type: DataTypes.STRING(20),
    allowNull: false
  },
  nps_score: {
    type: DataTypes.INTEGER,
    allowNull: false,
    validate: {
      min: 0,
      max: 10
    }
  },
  content_satisfaction: {
    type: DataTypes.INTEGER,
    allowNull: false,
    validate: {
      min: 1,
      max: 5
    }
  },
  value_perception: {
    type: DataTypes.STRING(20),
    allowNull: false
  },
  churn_likelihood: {
    type: DataTypes.STRING(20),
    allowNull: false
  },
  switching_behavior: {
    type: DataTypes.STRING(50),
    allowNull: false
  },
  switching_reason: {
    type: DataTypes.TEXT,
    allowNull: true
  },
  account_sharing: {
    type: DataTypes.STRING(50),
    allowNull: false
  },
  competitor_awareness: {
    type: DataTypes.STRING(50),
    allowNull: false
  },
  bundling_preference: {
    type: DataTypes.STRING(50),
    allowNull: false
  },
  payment_method: {
    type: DataTypes.STRING(50),
    allowNull: false
  },
  suggestions: {
    type: DataTypes.TEXT,
    allowNull: true
  },
  predicted_churn_probability: {
    type: DataTypes.FLOAT,
    allowNull: true
  },
  churn_risk_category: {
    type: DataTypes.STRING(20),
    allowNull: true
  }
}, {
  tableName: 'customers',
  timestamps: true,
  createdAt: 'created_at',
  updatedAt: 'updated_at'
});

// Subscription Model (for multiple platforms)
const Subscription = sequelize.define('Subscription', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  platform_name: {
    type: DataTypes.STRING(50),
    allowNull: false
  },
  is_primary: {
    type: DataTypes.BOOLEAN,
    defaultValue: false
  }
}, {
  tableName: 'subscriptions',
  timestamps: true
});

// Define relationships
Customer.hasMany(Subscription, { foreignKey: 'customer_id', as: 'subscriptions' });
Subscription.belongsTo(Customer, { foreignKey: 'customer_id', as: 'customer' });

// Sync models with database (create tables)
const syncDatabase = async (force = false) => {
  try {
    await sequelize.sync({ force });
    console.log('✅ Database tables created/updated successfully.');
    return true;
  } catch (error) {
    console.error('❌ Error syncing database:', error.message);
    return false;
  }
};

module.exports = {
  Customer,
  Subscription,
  syncDatabase
};