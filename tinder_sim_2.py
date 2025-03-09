import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def run_tinder_simulation(n_men=100, n_women=100, days=30, daily_swipes=10, 
                          men_right_swipe_pct=50, women_right_swipe_pct=20,
                          stop_swiping_prob=0.5, inactive_days=7,
                          match_threshold=3, threshold_days=7, penalty_factor=0.5):
    """
    Simulate a Tinder-like matching system with variable population size, inactive periods,
    and penalties for users who don't get enough matches early on.
    
    Parameters:
    - n_men: Number of men in the population
    - n_women: Number of women in the population
    - days: Number of days to run the simulation
    - daily_swipes: Number of daily swipes allowed
    - men_right_swipe_pct: Percentage of right swipes by men
    - women_right_swipe_pct: Percentage of right swipes by women
    - stop_swiping_prob: Probability to stop swiping after 5 matches
    - inactive_days: Number of days a person is inactive after deciding to stop
    - match_threshold: Minimum matches needed in first threshold_days
    - threshold_days: Number of days to reach match_threshold
    - penalty_factor: Factor to reduce right swipe probability for users below threshold
    
    Returns:
    - Dictionaries of matches per person and activity stats
    """
    # Initialize match counters
    men_matches = {i: 0 for i in range(n_men)}
    women_matches = {i: 0 for i in range(n_women)}
    
    # Track inactive status (days remaining inactive)
    men_inactive = {i: 0 for i in range(n_men)}
    women_inactive = {i: 0 for i in range(n_women)}
    
    # Track if penalty has been applied (to avoid applying multiple times)
    men_penalized = {i: False for i in range(n_men)}
    women_penalized = {i: False for i in range(n_women)}
    
    # Track match history for threshold check
    men_match_history = {i: {} for i in range(n_men)}
    women_match_history = {i: {} for i in range(n_women)}
    
    # Track daily activity stats
    daily_active_men = []
    daily_active_women = []
    daily_penalty_men = []
    daily_penalty_women = []
    
    # Initial probabilities of swiping right
    men_prob = men_right_swipe_pct / 100
    women_prob = women_right_swipe_pct / 100
    
    # Individual probabilities (may be reduced later)
    men_individual_prob = {i: men_prob for i in range(n_men)}
    women_individual_prob = {i: women_prob for i in range(n_women)}
    
    # Run simulation for specified number of days
    for day in range(days):
        # Count active users for this day
        active_men = sum(1 for i in range(n_men) if men_inactive[i] == 0)
        active_women = sum(1 for i in range(n_women) if women_inactive[i] == 0)
        
        # Count penalized users
        penalized_men = sum(1 for i in range(n_men) if men_penalized[i])
        penalized_women = sum(1 for i in range(n_women) if women_penalized[i])
        
        daily_active_men.append(active_men)
        daily_active_women.append(active_women)
        daily_penalty_men.append(penalized_men)
        daily_penalty_women.append(penalized_women)
        
        # Apply penalties if threshold day has been reached
        if day == threshold_days:
            for i in range(n_men):
                # Count matches in first X days
                early_matches = men_matches[i]
                if early_matches < match_threshold:
                    men_individual_prob[i] *= penalty_factor
                    men_penalized[i] = True
                    
            for i in range(n_women):
                # Count matches in first X days
                early_matches = women_matches[i]
                if early_matches < match_threshold:
                    women_individual_prob[i] *= penalty_factor
                    women_penalized[i] = True
        
        # Each active man gets daily_swipes per day
        for man_id in range(n_men):
            # Skip if man is inactive
            if men_inactive[man_id] > 0:
                men_inactive[man_id] -= 1
                continue
                
            # Get list of active women
            active_women_ids = [i for i in range(n_women) if women_inactive[i] == 0]
            
            # If no active women, skip this man's turn
            if not active_women_ids:
                continue
                
            # Randomly select women to swipe on (from active pool)
            swipes_available = min(daily_swipes, len(active_women_ids))
            if swipes_available == 0:
                continue
                
            women_to_swipe = np.random.choice(active_women_ids, swipes_available, replace=False)
            
            day_matches = 0
            for woman_id in women_to_swipe:
                # Determine if both swipe right (using individual probabilities)
                man_swipes_right = np.random.random() < men_individual_prob[man_id]
                woman_swipes_right = np.random.random() < women_individual_prob[woman_id]
                
                # If both swipe right, it's a match
                if man_swipes_right and woman_swipes_right:
                    men_matches[man_id] += 1
                    women_matches[woman_id] += 1
                    
                    # Record match day for history
                    men_match_history[man_id][day] = men_match_history[man_id].get(day, 0) + 1
                    women_match_history[woman_id][day] = women_match_history[woman_id].get(day, 0) + 1
                    
                    day_matches += 1
                    
                    # Check if user stops swiping after 5 matches (total, not just today)
                    if men_matches[man_id] >= 5:
                        if np.random.random() < stop_swiping_prob:
                            men_inactive[man_id] = inactive_days
                            break
                            
                    if women_matches[woman_id] >= 5:
                        if np.random.random() < stop_swiping_prob:
                            women_inactive[woman_id] = inactive_days
        
        # Each active woman gets daily_swipes per day (for women who haven't been swiped on yet)
        for woman_id in range(n_women):
            # Skip if woman is inactive
            if women_inactive[woman_id] > 0:
                women_inactive[woman_id] -= 1
                continue
                
            # Get list of active men
            active_men_ids = [i for i in range(n_men) if men_inactive[i] == 0]
            
            # If no active men, skip this woman's turn
            if not active_men_ids:
                continue
                
            # Randomly select men to swipe on (from active pool)
            swipes_available = min(daily_swipes, len(active_men_ids))
            if swipes_available == 0:
                continue
                
            men_to_swipe = np.random.choice(active_men_ids, swipes_available, replace=False)
            
            day_matches = 0
            for man_id in men_to_swipe:
                # Determine if both swipe right (using individual probabilities)
                man_swipes_right = np.random.random() < men_individual_prob[man_id]
                woman_swipes_right = np.random.random() < women_individual_prob[woman_id]
                
                # If both swipe right, it's a match
                if man_swipes_right and woman_swipes_right:
                    men_matches[man_id] += 1
                    women_matches[woman_id] += 1
                    
                    # Record match day for history
                    men_match_history[man_id][day] = men_match_history[man_id].get(day, 0) + 1
                    women_match_history[woman_id][day] = women_match_history[woman_id].get(day, 0) + 1
                    
                    day_matches += 1
                    
                    # Check if user stops swiping after 5 matches (total, not just today)
                    if men_matches[man_id] >= 5:
                        if np.random.random() < stop_swiping_prob:
                            men_inactive[man_id] = inactive_days
                            
                    if women_matches[woman_id] >= 5:
                        if np.random.random() < stop_swiping_prob:
                            women_inactive[woman_id] = inactive_days
                            break
    
    # Combine all matches
    all_matches = list(men_matches.values()) + list(women_matches.values())
    
    activity_stats = {
        'daily_active_men': daily_active_men,
        'daily_active_women': daily_active_women,
        'daily_penalty_men': daily_penalty_men,
        'daily_penalty_women': daily_penalty_women
    }
    
    penalty_stats = {
        'men_penalized': sum(men_penalized.values()),
        'women_penalized': sum(women_penalized.values())
    }
    
    return men_matches, women_matches, all_matches, activity_stats, penalty_stats

def plot_match_histogram(men_matches, women_matches, all_matches, n_men, n_women):
    """Create histograms for matches"""
    
    # Convert to DataFrames for easier plotting
    men_df = pd.DataFrame({
        'Matches': list(men_matches.values()),
        'Gender': ['Men'] * len(men_matches)
    })
    
    women_df = pd.DataFrame({
        'Matches': list(women_matches.values()),
        'Gender': ['Women'] * len(women_matches)
    })
    
    all_df = pd.DataFrame({
        'Matches': all_matches,
        'Gender': ['Men'] * len(men_matches) + ['Women'] * len(women_matches)
    })
    
    # Calculate percentages for each match count by gender
    men_counts = Counter(men_df['Matches'])
    women_counts = Counter(women_df['Matches'])
    all_counts = Counter(all_df['Matches'])
    
    men_pct = {k: v/n_men*100 for k, v in men_counts.items()}
    women_pct = {k: v/n_women*100 for k, v in women_counts.items()}
    all_pct = {k: v/(n_men+n_women)*100 for k, v in all_counts.items()}
    
    # Create DataFrames for percentage plots
    men_pct_df = pd.DataFrame({
        'Matches': list(men_pct.keys()),
        'Percentage': list(men_pct.values()),
        'Gender': ['Men'] * len(men_pct)
    })
    
    women_pct_df = pd.DataFrame({
        'Matches': list(women_pct.keys()),
        'Percentage': list(women_pct.values()),
        'Gender': ['Women'] * len(women_pct)
    })
    
    all_pct_df = pd.concat([men_pct_df, women_pct_df])
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Separate histograms for men and women
    sns.barplot(x='Matches', y='Percentage', hue='Gender', data=all_pct_df, ax=axes[0])
    axes[0].set_title('Match Distribution by Gender')
    axes[0].set_xlabel('Number of Matches')
    axes[0].set_ylabel('Percentage of Population')
    
    # Plot 2: Overall histogram
    all_pct_df_overall = pd.DataFrame({
        'Matches': list(all_pct.keys()),
        'Percentage': list(all_pct.values())
    })
    
    sns.barplot(x='Matches', y='Percentage', data=all_pct_df_overall, color='purple', ax=axes[1])
    axes[1].set_title('Overall Match Distribution')
    axes[1].set_xlabel('Number of Matches')
    axes[1].set_ylabel('Percentage of Population')
    
    plt.tight_layout()
    return fig

def plot_activity(activity_stats, days, match_threshold_day):
    """Plot the active users and penalized users over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    days_range = list(range(1, days + 1))
    
    # Top plot: Active users
    ax1.plot(days_range, activity_stats['daily_active_men'], label='Active Men')
    ax1.plot(days_range, activity_stats['daily_active_women'], label='Active Women')
    
    # Add vertical line at threshold day
    if match_threshold_day < days:
        ax1.axvline(x=match_threshold_day + 1, color='r', linestyle='--', alpha=0.7, 
                   label=f'Match Threshold Day ({match_threshold_day})')
    
    ax1.set_title('Active Users Over Time')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Number of Active Users')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Bottom plot: Penalized users
    ax2.plot(days_range, activity_stats['daily_penalty_men'], label='Penalized Men')
    ax2.plot(days_range, activity_stats['daily_penalty_women'], label='Penalized Women')
    
    # Add vertical line at threshold day
    if match_threshold_day < days:
        ax2.axvline(x=match_threshold_day + 1, color='r', linestyle='--', alpha=0.7,
                   label=f'Match Threshold Day ({match_threshold_day})')
    
    ax2.set_title('Penalized Users Over Time')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Number of Penalized Users')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

# Streamlit app
st.title("Tinder Matching Simulation")

st.sidebar.header("Population Parameters")
n_men = st.sidebar.slider("Number of Men", 10, 500, 100)
n_women = st.sidebar.slider("Number of Women", 10, 500, 100)

st.sidebar.header("Swiping Behavior")
men_right_swipe_pct = st.sidebar.slider("% Right Swipes from Men", 0, 100, 50)
women_right_swipe_pct = st.sidebar.slider("% Right Swipes from Women", 0, 100, 20)
daily_swipes = st.sidebar.slider("Daily Swipes Allowed", 1, 50, 10)

st.sidebar.header("Activity Parameters")
stop_swiping_prob = st.sidebar.slider("Probability to Stop After 5+ Matches", 0.0, 1.0, 0.5)
inactive_days = st.sidebar.slider("Inactive Period (Days)", 1, 30, 7)
days = st.sidebar.slider("Total Days to Simulate", 1, 100, 30)

st.sidebar.header("Match Threshold Parameters")
match_threshold = st.sidebar.slider("Minimum Matches Needed (Y)", 1, 10, 3)
threshold_days = st.sidebar.slider("Days to Reach Threshold (X)", 1, 50, 7)
penalty_factor = st.sidebar.slider("Match Probability Reduction Factor (Z)", 0.1, 1.0, 0.5, 0.1)

if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        men_matches, women_matches, all_matches, activity_stats, penalty_stats = run_tinder_simulation(
            n_men=n_men,
            n_women=n_women,
            days=days,
            daily_swipes=daily_swipes,
            men_right_swipe_pct=men_right_swipe_pct,
            women_right_swipe_pct=women_right_swipe_pct,
            stop_swiping_prob=stop_swiping_prob,
            inactive_days=inactive_days,
            match_threshold=match_threshold,
            threshold_days=threshold_days,
            penalty_factor=penalty_factor
        )
        
        # Display statistics
        st.subheader("Simulation Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg. Matches (Men)", f"{np.mean(list(men_matches.values())):.1f}")
        
        with col2:
            st.metric("Avg. Matches (Women)", f"{np.mean(list(women_matches.values())):.1f}")
        
        with col3:
            st.metric("Avg. Matches (Overall)", f"{np.mean(all_matches):.1f}")
        
        # Plot the activity over time
        st.subheader("User Activity and Penalties Over Time")
        activity_fig = plot_activity(activity_stats, days, threshold_days)
        st.pyplot(activity_fig)
        
        # Display penalty statistics
        st.subheader("Penalty Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Men Penalized", f"{penalty_stats['men_penalized']} / {n_men}")
            st.metric("Men Penalized %", f"{penalty_stats['men_penalized'] / n_men * 100:.1f}%")
        
        with col2:
            st.metric("Women Penalized", f"{penalty_stats['women_penalized']} / {n_women}")
            st.metric("Women Penalized %", f"{penalty_stats['women_penalized'] / n_women * 100:.1f}%")
        
        # Plot the histograms
        st.subheader("Match Distribution")
        match_fig = plot_match_histogram(men_matches, women_matches, all_matches, n_men, n_women)
        st.pyplot(match_fig)
        
        # Display match distribution tables
        st.subheader("Detailed Match Distribution")
        
        men_df = pd.DataFrame(list(men_matches.values()), columns=['Matches'])
        women_df = pd.DataFrame(list(women_matches.values()), columns=['Matches'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Men's Matches Distribution")
            men_counts = men_df['Matches'].value_counts().sort_index().reset_index()
            men_counts.columns = ['Match Count', 'Number of Men']
            men_counts['Percentage'] = men_counts['Number of Men'] / n_men * 100
            st.dataframe(men_counts)
        
        with col2:
            st.write("Women's Matches Distribution")
            women_counts = women_df['Matches'].value_counts().sort_index().reset_index()
            women_counts.columns = ['Match Count', 'Number of Women']
            women_counts['Percentage'] = women_counts['Number of Women'] / n_women * 100
            st.dataframe(women_counts)
        
        # Add more detailed statistics
        st.subheader("Additional Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Men's Statistics")
            st.metric("Zero Matches %", f"{(sum(1 for m in men_matches.values() if m == 0) / n_men * 100):.1f}%")
            st.metric("5+ Matches %", f"{(sum(1 for m in men_matches.values() if m >= 5) / n_men * 100):.1f}%")
            st.metric("Maximum Matches", max(men_matches.values()))
        
        with col2:
            st.write("Women's Statistics")
            st.metric("Zero Matches %", f"{(sum(1 for w in women_matches.values() if w == 0) / n_women * 100):.1f}%")
            st.metric("5+ Matches %", f"{(sum(1 for w in women_matches.values() if w >= 5) / n_women * 100):.1f}%")
            st.metric("Maximum Matches", max(women_matches.values()))
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to start.")

# Initial instructions
st.markdown("""
### Instructions
1. Use the sliders in the sidebar to set the simulation parameters
2. Click 'Run Simulation' to see the results
3. The histograms show the percentage of population by number of matches
4. The activity graphs show active users and penalized users over time

### Match Threshold Parameters
- **Minimum Matches Needed (Y)**: Users must get at least this many matches in the first X days
- **Days to Reach Threshold (X)**: Time period to evaluate early matching success
- **Match Probability Reduction Factor (Z)**: How much to reduce matching probability for users below threshold
  - Example: If Z = 0.5, users who don't reach the threshold will have their right-swipe probability cut in half

### Behavior Notes
- If a user doesn't get Y matches within the first X days, their probability of receiving right swipes decreases by factor Z
- This simulates how dating apps might reduce visibility for less popular profiles
- The vertical red line in the graphs marks when the threshold is evaluated and penalties are applied
- The bottom graph shows how many users are penalized over time
""")
