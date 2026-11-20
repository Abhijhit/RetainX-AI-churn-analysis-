import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sqlite3
import os

# Create charts directory
os.makedirs('static/charts', exist_ok=True)

print("Starting chart generation from database...")

# Connect to database
conn = sqlite3.connect('churn_data.db')
cursor = conn.cursor()

try:
    # Get risk level distribution
    cursor.execute("SELECT churn_risk_category, COUNT(*) FROM customers GROUP BY churn_risk_category")
    risk_data = cursor.fetchall()
    print("Risk data:", risk_data)
    
    if risk_data:
        labels = [row[0] for row in risk_data]
        sizes = [row[1] for row in risk_data]
        colors = ['#e74c3c', '#f39c12', '#2ecc71']
        
        # Pie Chart
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Customer Risk Level Distribution')
        plt.savefig('static/charts/risk_pie.png')
        plt.close()
        print("Saved risk_pie.png")
        
        # Bar Chart
        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, sizes, color=colors)
        plt.xlabel('Risk Level')
        plt.ylabel('Number of Customers')
        plt.title('Risk Level Distribution')
        for bar, size in zip(bars, sizes):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(size), ha='center')
        plt.savefig('static/charts/risk_bar.png')
        plt.close()
        print("Saved risk_bar.png")
    
    # Get age group distribution
    cursor.execute("SELECT age_group, COUNT(*) FROM customers WHERE age_group IS NOT NULL GROUP BY age_group")
    age_data = cursor.fetchall()
    if age_data:
        ages = [row[0] for row in age_data]
        counts = [row[1] for row in age_data]
        plt.figure(figsize=(10, 6))
        plt.bar(ages, counts, color='#3498db')
        plt.xlabel('Age Group')
        plt.ylabel('Number of Customers')
        plt.title('Customer Distribution by Age Group')
        plt.xticks(rotation=45)
        for bar, count in zip(plt.gca().patches, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(count), ha='center')
        plt.tight_layout()
        plt.savefig('static/charts/age_group.png')
        plt.close()
        print("Saved age_group.png")
    
    # Get subscription duration distribution
    cursor.execute("SELECT subscription_duration, COUNT(*) FROM customers WHERE subscription_duration IS NOT NULL GROUP BY subscription_duration")
    duration_data = cursor.fetchall()
    if duration_data:
        durations = [row[0] for row in duration_data]
        counts = [row[1] for row in duration_data]
        plt.figure(figsize=(10, 6))
        plt.bar(durations, counts, color='#9b59b6')
        plt.xlabel('Subscription Duration')
        plt.ylabel('Number of Customers')
        plt.title('Customer Distribution by Subscription Duration')
        plt.xticks(rotation=45)
        for bar, count in zip(plt.gca().patches, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(count), ha='center')
        plt.tight_layout()
        plt.savefig('static/charts/duration.png')
        plt.close()
        print("Saved duration.png")
    
    # Get churn probability histogram
    cursor.execute("SELECT predicted_churn_probability FROM customers WHERE predicted_churn_probability IS NOT NULL")
    probabilities = [row[0] for row in cursor.fetchall()]
    if probabilities:
        plt.figure(figsize=(8, 6))
        plt.hist(probabilities, bins=20, color='#e67e22', edgecolor='black')
        plt.xlabel('Churn Probability')
        plt.ylabel('Number of Customers')
        plt.title('Distribution of Churn Probabilities')
        plt.axvline(x=0.4, color='orange', linestyle='--', label='Medium Risk (0.4)')
        plt.axvline(x=0.7, color='red', linestyle='--', label='High Risk (0.7)')
        plt.legend()
        plt.savefig('static/charts/probability_hist.png')
        plt.close()
        print("Saved probability_hist.png")
    
    print("All charts generated successfully!")
    
except Exception as e:
    print("Error:", e)

conn.close()