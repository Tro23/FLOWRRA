"""
FLOWRRA: Exploration Swarm MVP
Run this file to train/simulate the swarm.
"""
from flowrra.core import FLOWRRA_Orchestrator
from flowrra.config import CONFIG
from utils.blender_export import save_history_for_blender
import matplotlib.pyplot as plt 

def main():
    print("Initializing FLOWRRA Exploration Swarm...")
    sim = FLOWRRA_Orchestrator()
    
    steps = 500
    rewards_log = []
    
    print(f"Target: Explore {CONFIG['spatial']['world_bounds']} area.")
    
    try:
        for t in range(steps):
            avg_reward = sim.step(t)
            rewards_log.append(avg_reward)
            
            cov = sim.map.get_coverage_percentage()
            
            print(f"\rStep {t}/{steps} | Coverage: {cov:.2f}% | Reward: {avg_reward:.4f}", end="")
            
            if cov > 95.0:
                print("\nMission Complete: >95% Coverage achieved.")
                break
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        
    # Save Result
    save_history_for_blender(sim.history, "flowrra_viz.json")
    
    # Plot Training/Performance
    plt.plot(rewards_log)
    plt.title("Swarm Average Reward over Time")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.show()

if __name__ == "__main__":
    main()