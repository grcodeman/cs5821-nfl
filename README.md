# CS5821 NFL Over/Under Predictor

This project uses a Random Forest regressor to predict the total number of points scored in an NFL game, given two teams, an over/under line, and a point spread.

## How to Run

1. **Install Requirements:**  
   Make sure you have all necessary dependencies installed.  
   *Example:*  
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Program:**  
   Execute the `main.py` script:  
   ```bash
   python main.py
   ```

3. **Input Prompts:**  
   When prompted, enter the following information:
   - **Home Team:** (e.g., Detroit Lions)
   - **Away Team:** (e.g., Buffalo Bills)
   - **Over/Under (O/U):** (e.g., 50.5)
   - **Spread:** (e.g., 1.5)

   The program will use the provided inputs, along with historical data and a pre-trained Random Forest regressor (loaded from a pickle file), to predict the total number of points scored.
