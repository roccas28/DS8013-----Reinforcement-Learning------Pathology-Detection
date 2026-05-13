import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
import cv2
import matplotlib.patches as patches

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12, 
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

def plot_training_rewards(csv_path='training_metrics.csv'): # Plots Cumulative Reward
        
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Episode", y="Reward", color="b", alpha=0.6)
    
    df['Moving_Avg'] = df['Reward'].rolling(window=50, min_periods=1).mean()
    sns.lineplot(data=df, x="Episode", y="Moving_Avg", color="r", linewidth=2, label="50-Ep Moving Avg")
    
    plt.title("RL Agent Training Curve: Cumulative Reward per Episode")
    plt.ylabel("Total Reward")
    plt.xlabel("Training Episode")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("plot_training_rewards.png", dpi=300)
    print("Saved plot_training_rewards.png")

def plot_training_steps(csv_path='training_metrics.csv'): # Plots Efficiency (Steps to Diagnose)

    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Episode", y="Steps", color="g", alpha=0.6)
    
    df['Moving_Avg'] = df['Steps'].rolling(window=50, min_periods=1).mean()
    sns.lineplot(data=df, x="Episode", y="Moving_Avg", color="darkgreen", linewidth=2, label="50-Ep Moving Avg")
    
    plt.title("Agent Efficiency: Steps Taken Before Diagnosis")
    plt.ylabel("Number of Glimpses (Steps)")
    plt.xlabel("Training Episode")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("plot_training_steps.png", dpi=300)
    print("Saved plot_training_steps.png")

def plot_training_loss(csv_path='training_metrics.csv'): # Plots TD Loss (Network Convergence)

    df = pd.read_csv(csv_path)
    # Drop episodes where loss might be 0/NaN (early exploration before replay buffer fills)
    df = df[df['Loss'] > 0] 
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Episode", y="Loss", color="purple", alpha=0.4)
    
    df['Moving_Avg'] = df['Loss'].rolling(window=50, min_periods=1).mean()
    sns.lineplot(data=df, x="Episode", y="Moving_Avg", color="indigo", linewidth=2, label="50-Ep Moving Avg")
    
    plt.title("DDQN Optimization: Temporal Difference (TD) Loss")
    plt.ylabel("Huber Loss")
    plt.xlabel("Training Episode")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("plot_training_loss.png", dpi=300)
    print("Saved plot_training_loss.png")

def plot_validation_metrics(csv_path='validation_metrics.csv'): # Plots F1 and AUC

    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=df, x="Episode", y="F1", marker="o", color="blue", label="F1-Score", linewidth=2)
    sns.lineplot(data=df, x="Episode", y="AUC", marker="s", color="orange", label="AUC-ROC", linewidth=2)
    
    plt.title("Clinical Validation Performance on Unseen X-Rays")
    plt.ylabel("Score (0.0 to 1.0)")
    plt.xlabel("Training Episode")
    plt.ylim(0, 1.05)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("plot_validation_metrics.png", dpi=300)
    print("Saved plot_validation_metrics.png")

def plot_confusion_matrix(csv_path='last_validation_preds.csv'): # Confusion Matrix via last validation pred

    df = pd.read_csv(csv_path)
    cm = confusion_matrix(df['True_Label'], df['Predicted_Label'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Diagnosed Negative', 'Diagnosed Positive'], 
                yticklabels=['Actual Negative', 'Actual Positive'])
    
    plt.title("Clinical Confusion Matrix (Final Validation Set)")
    plt.ylabel("True Pathology")
    plt.xlabel("Agent Diagnosis")
    plt.tight_layout()
    plt.savefig("plot_confusion_matrix.png", dpi=300)
    print("Saved plot_confusion_matrix.png")

def plot_agent_trajectory(csv_path='trajectory_data.csv'): # Trajectory path 

    df = pd.read_csv(csv_path)
    img_path = df['Image_Path'].iloc[0]
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    img = cv2.resize(img, (224, 224))
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    
    # Plot the path the agent took
    xs = df['X'].tolist()
    ys = df['Y'].tolist()
    
    plt.plot(xs, ys, color='red', marker='o', linewidth=2, markersize=4, label='Agent Path')
    plt.plot(xs[0], ys[0], color='blue', marker='s', markersize=8, label='Start (Center)')
    plt.plot(xs[-1], ys[-1], color='green', marker='*', markersize=12, label='Final Diagnosis Point')
    
    # Draw a box representing the final glimpse crop (84x84)
    rect = patches.Rectangle((xs[-1] - 42, ys[-1] - 42), 84, 84, linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--', label='Final Glimpse')
    plt.gca().add_patch(rect)
    
    actual = "Positive" if df['Actual_Label'].iloc[0] == 1 else "Negative"
    guess = "Positive" if df['Predicted_Label'].iloc[0] == 1 else "Negative"
    
    plt.title(f"Agent Spatial Trajectory\nActual: {actual} | Diagnosed: {guess}")
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("plot_agent_trajectory.png", dpi=300)
    print("Saved plot_agent_trajectory.png")

def generate_all_plots(): # Master Function
    print("\nGenerating Report Visualizations...")
    plot_training_rewards('training_metrics.csv')
    plot_training_steps('training_metrics.csv')
    plot_training_loss('training_metrics.csv')
    plot_validation_metrics('validation_metrics.csv')
    plot_confusion_matrix('last_validation_preds.csv')
    plot_agent_trajectory('trajectory_data.csv')
