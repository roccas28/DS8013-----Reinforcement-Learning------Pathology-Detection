import pandas as pd
import numpy as np
import time

# Custom Modules
import utils as ut
import env as environment
import train as tr
import plots as pt

def run():
    print("DS8013 Project: RL-Based Visual Attention Model for Pathology Detection")

    data_list = ut.load_chexpert_labels('train.csv') 
    valid_list = ut.load_chexpert_labels('valid.csv') # Load Validation Data
    
    if not data_list or not valid_list:
        print("Data lists empty. Please verify data folder.")
        return

    env = environment.XRayAttentionEnv(data_list)
    valid_env = environment.XRayAttentionEnv(valid_list) # Separate Validation Env
    
    agent = tr.DQNTrainer(action_size=int(env.action_space.n))
    
    episodes = 15000
    metrics_history = [] 
    validation_history = [] # Track F1 and AUC
    
    print("\nStarting Training Loop...")
    start_time = time.time()

    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        episode_losses = []
        done = False
        
        # --- TRAINING PHASE ---
        while not done:
            action = agent.act(state, explore=True)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            
            loss = agent.replay() 
            if loss is not None:
                episode_losses.append(loss)

        # Decay epsilon exactly once per episode
        agent.decay_epsilon()
        
        if e % 10 == 0:
            agent.update_target_network()
            print(f"Episode: {e}/{episodes} | Reward: {total_reward:.2f} | Steps: {steps} | Epsilon: {agent.epsilon:.2f}")
            
        avg_loss = sum(episode_losses)/len(episode_losses) if episode_losses else 0.0
        metrics_history.append([e, total_reward, steps, avg_loss])
        
        # --- VALIDATION PHASE (Every 500 Episodes) ---
        if e > 0 and (e+1) % 500 == 0:
            print(f"\n--- Running Validation on 200 Images ---")
            y_true = []
            y_pred = []
            
            for v in range(len(valid_list)):
                v_state, _ = valid_env.reset()
                v_done = False
                
                while not v_done:
                    v_action = agent.act(v_state, explore=False) # STRICT EXPLOITATION
                    v_next_state, _, v_done, _, v_info = valid_env.step(v_action)
                    v_state = v_next_state
                    
                diagnosis = v_info.get('diagnosis', 0)
                if diagnosis == -1: diagnosis = 0 # Default to negative if it timed out
                
                y_true.append(valid_env.label)
                y_pred.append(diagnosis)
                
            f1, auc = ut.calculate_metrics(y_true, y_pred, y_pred) # Pass pred as prob proxy
            print(f"Validation F1-Score: {f1:.3f} | AUC-ROC: {auc:.3f}\n")

            # Save raw predictions for the Confusion Matrix
            pd.DataFrame({'True_Label': y_true, 'Predicted_Label': y_pred}).to_csv("last_validation_preds.csv", index=False)
            
            validation_history.append([e, f1, auc])
            pd.DataFrame(validation_history, columns=["Episode", "F1", "AUC"]).to_csv("validation_metrics.csv", index=False)

            checkpoint_path = f"saved_models/dqn_weights_ep{e+1}.weights.h5" # Save model weights (in case of crash - Windows Updates)
            agent.main_network.save_weights(checkpoint_path)
            print(f"Model weights saved to {checkpoint_path}")
        
    print(f"\nTraining Complete! Time elapsed: {(time.time() - start_time) / 60:.2f} minutes")

    # --- END OF TRAINING: EXTRACT SAMPLE TRAJECTORY ---
    print("\nExtracting Sample Spatial Trajectory...")
    t_state, _ = valid_env.reset()
    t_done = False
    trajectory_coords = []
    
    # Record starting center
    trajectory_coords.append([valid_env.current_img_path, valid_env.label, -1, 112, 112]) 

    while not t_done:
        t_action = agent.act(t_state, explore=False) # Pure exploitation
        t_next_state, _, t_done, _, t_info = valid_env.step(t_action)
        t_state = t_next_state
        
        x = t_info.get('x', 112)
        y = t_info.get('y', 112)
        diagnosis = t_info.get('diagnosis', -1)
        trajectory_coords.append([valid_env.current_img_path, valid_env.label, diagnosis, x, y])

    # Save to CSV so plots.py can map it
    pd.DataFrame(trajectory_coords, columns=["Image_Path", "Actual_Label", "Predicted_Label", "X", "Y"]).to_csv("trajectory_data.csv", index=False)
    
    cols = ["Episode", "Reward", "Steps", "Loss"]
    pd.DataFrame(metrics_history, columns=cols).to_csv("training_metrics.csv", index=False)
    pt.generate_all_plots()

if __name__ == "__main__":
    run()
