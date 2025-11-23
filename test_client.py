"""
Test client for the Marketing Agent API
Run this script to test the agent with sample data
"""

import requests
import pandas as pd
import io
import json

# API endpoint
BASE_URL = "http://localhost:8000"


def generate_sample_data():
    """Generate sample ad performance CSV"""
    sample_data = {
        "campaign_name": [
            "Summer Sale - Video Ads",
            "Spring Collection - Carousel",
            "General Awareness - Static",
            "Retargeting - Dynamic",
            "New Product Launch - Story",
            "Holiday Special - Video",
            "Brand Campaign - Display",
            "Promo Code - Search",
            "Flash Sale - Instagram",
            "Bundle Offer - Facebook",
            "Seasonal Campaign - YouTube",
            "Clearance Sale - Google Display"
        ],
        "impressions": [125000, 89000, 210000, 45000, 67000, 98000, 156000, 34000, 
                       178000, 92000, 145000, 67000],
        "clicks": [6000, 1780, 3150, 2250, 3015, 4410, 2340, 1870,
                  7120, 2300, 5800, 1340],
        "conversions": [288, 71, 63, 180, 151, 220, 47, 150,
                       356, 115, 290, 67],
        "spend": [4500, 3200, 5100, 2800, 3400, 4200, 4800, 2500,
                 5600, 3800, 6200, 2900]
    }
    
    df = pd.DataFrame(sample_data)
    return df


def test_health_check():
    """Test the health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_run_agent():
    """Test the main agent endpoint with sample data"""
    print("\n" + "="*60)
    print("Testing Agent Analysis")
    print("="*60)
    
    # Generate sample data
    df = generate_sample_data()
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    # Prepare file upload
    files = {
        'file': ('test_data.csv', csv_content, 'text/csv')
    }
    
    # Make request
    response = requests.post(f"{BASE_URL}/run-agent", files=files)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n--- Performance Summary ---")
        summary = result['performance_analysis']['summary']
        print(f"Total Campaigns: {summary['total_campaigns']}")
        print(f"Total Spend: ${summary['total_spend']:,.2f}")
        print(f"Average CTR: {summary['avg_ctr']:.2f}%")
        print(f"Average CVR: {summary['avg_cvr']:.2f}%")
        print(f"Average CPC: ${summary['avg_cpc']:.2f}")
        
        print("\n--- Top Performers ---")
        for campaign in result['performance_analysis']['top_performers'][:3]:
            print(f"  • {campaign['campaign_name']}: CTR {campaign['ctr']:.2f}%, CVR {campaign['cvr']:.2f}%")
        
        print("\n--- Graph RAG Insights ---")
        for insight in result['graph_insights'][:5]:
            print(f"  • {insight}")
        
        print("\n--- Creative Recommendations ---")
        for rec in result['creative_recommendations'][:3]:
            print(f"\n  [{rec['priority'].upper()}] {rec['category']}")
            print(f"  Suggestion: {rec['suggestion']}")
            print(f"  Expected Lift: {rec['expected_lift']}")
            print(f"  Reasoning: {rec['reasoning']}")
        
        print("\n--- Evaluation Metrics ---")
        metrics = result['evaluation_metrics']
        print(f"  Total Analyses: {metrics.get('total_analyses', 0)}")
        print(f"  Knowledge Graph Nodes: {metrics.get('knowledge_graph_nodes', 0)}")
        print(f"  Knowledge Graph Edges: {metrics.get('knowledge_graph_edges', 0)}")
        print(f"  Relevance Score: {metrics.get('relevance_score', 0):.2f}")
        print(f"  Hallucination Rate: {metrics.get('hallucination_rate', 0):.2%}")
        
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_metrics():
    """Test the metrics endpoint"""
    print("\n" + "="*60)
    print("Testing Metrics Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"Status Code: {response.status_code}")
    print(f"Metrics: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_knowledge_graph():
    """Test the knowledge graph endpoint"""
    print("\n" + "="*60)
    print("Testing Knowledge Graph Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/knowledge-graph")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        kg = response.json()
        print(f"Total Nodes: {len(kg['nodes'])}")
        print(f"Total Edge Connections: {kg['edge_count']}")
        print("\nSample Nodes:")
        for node_id, attributes in list(kg['nodes'].items())[:3]:
            print(f"  • {node_id}: {attributes}")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_feedback():
    """Test the feedback endpoint"""
    print("\n" + "="*60)
    print("Testing Feedback Endpoint")
    print("="*60)
    
    feedback_data = {
        "rating": 4,
        "comments": "Great insights, but could be more specific to my industry",
        "recommendation_id": "rec_001"
    }
    
    response = requests.post(f"{BASE_URL}/feedback", json=feedback_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def save_sample_csv():
    """Save a sample CSV file for manual testing"""
    df = generate_sample_data()
    filename = "sample_ad_performance.csv"
    df.to_csv(filename, index=False)
    print(f"\n✓ Sample CSV saved as: {filename}")
    print("You can test the API manually using:")
    print(f'  curl -X POST "{BASE_URL}/run-agent" -F "file=@{filename}"')


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("MARKETING AGENT API - TEST SUITE")
    print("="*80)
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print("Start it with: python main.py")
    
    input("\nPress Enter to start tests...")
    
    tests = [
        ("Health Check", test_health_check),
        ("Agent Analysis", test_run_agent),
        ("Metrics", test_metrics),
        ("Knowledge Graph", test_knowledge_graph),
        ("Feedback", test_feedback)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "✓ PASSED" if success else "✗ FAILED"))
        except requests.exceptions.ConnectionError:
            results.append((test_name, "✗ FAILED - Server not running"))
        except Exception as e:
            results.append((test_name, f"✗ FAILED - {str(e)}"))
    
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    for test_name, result in results:
        print(f"{test_name:.<40} {result}")
    
    # Save sample CSV
    save_sample_csv()
    
    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)


if __name__ == "__main__":
    run_all_tests()