using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
public class RLAgent: Agent{
    public override void OnEpisodeBegin(){
        // Reset the agent
    }

    public override void CollectObservations(VectorSensor sensor){
        // Collect observations
    }

    public override void OnActionReceived(float[] vectorAction){
        // Apply actions
    }

    public override void Heuristic(float[] actionsOut){
        // Define the heuristic
    }

}