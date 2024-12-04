import yaml
from src.portfolio.predictor import PortfolioPredictor

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize and run predictor
    predictor = PortfolioPredictor(config)
    predictor.run()

if __name__ == "__main__":
    main()
