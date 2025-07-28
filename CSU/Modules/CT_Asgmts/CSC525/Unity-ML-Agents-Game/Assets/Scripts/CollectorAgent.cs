using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class CollectorAgent : Agent
{
    [Header("Movement Settings")]
    public float moveSpeed = 5f;
    public float rotationSpeed = 300f;
    
    [Header("Environment References")]
    public Transform targetArea;
    public GameObject rewardItemPrefab;
    public GameObject penaltyItemPrefab;
    public Material winMaterial;
    public Material loseMaterial;
    public MeshRenderer floorMeshRenderer;
    
    [Header("Training Parameters")]
    public int maxItems = 10;
    public float episodeTimeout = 30f;
    
    private Rigidbody agentRb;
    private EnvironmentManager envManager;
    private Vector3 startingPos;
    private int itemsCollected = 0;
    private float episodeTimer = 0f;
    
    public override void Initialize()
    {
        agentRb = GetComponent<Rigidbody>();
        envManager = GetComponentInParent<EnvironmentManager>();
        startingPos = transform.localPosition;
    }
    
    public override void OnEpisodeBegin()
    {
        // Reset agent position and rotation
        transform.localPosition = startingPos;
        transform.localRotation = Quaternion.identity;
        agentRb.velocity = Vector3.zero;
        agentRb.angularVelocity = Vector3.zero;
        
        // Reset episode variables
        itemsCollected = 0;
        episodeTimer = 0f;
        
        // Reset environment
        if (envManager != null)
        {
            envManager.ResetEnvironment();
        }
        
        // Reset floor material
        if (floorMeshRenderer != null)
        {
            floorMeshRenderer.material = new Material(Shader.Find("Standard"));
            floorMeshRenderer.material.color = Color.white;
        }
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        // Agent's position and velocity (6 values)
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(agentRb.velocity);
        
        // Agent's rotation (4 values)
        sensor.AddObservation(transform.localRotation);
        
        // Distance to nearest reward and penalty items (2 values)
        sensor.AddObservation(GetDistanceToNearestItem("Reward"));
        sensor.AddObservation(GetDistanceToNearestItem("Penalty"));
        
        // Episode progress (1 value)
        sensor.AddObservation(episodeTimer / episodeTimeout);
        
        // Items collected ratio (1 value)
        sensor.AddObservation((float)itemsCollected / maxItems);
    }
    
    public override void OnActionReceived(ActionBuffers actions)
    {
        // Get continuous actions
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];
        
        // Apply movement
        Vector3 movement = new Vector3(moveX, 0, moveZ) * moveSpeed * Time.fixedDeltaTime;
        agentRb.MovePosition(transform.position + movement);
        
        // Update episode timer
        episodeTimer += Time.fixedDeltaTime;
        
        // Small negative reward for time to encourage efficiency
        AddReward(-0.001f);
        
        // Check for episode timeout
        if (episodeTimer >= episodeTimeout)
        {
            AddReward(-0.5f); // Penalty for timeout
            EndEpisode();
        }
        
        // Check if all items collected
        if (itemsCollected >= maxItems)
        {
            AddReward(2.0f); // Bonus for completing episode
            SetReward(GetCumulativeReward());
            if (floorMeshRenderer != null)
            {
                floorMeshRenderer.material = winMaterial;
            }
            EndEpisode();
        }
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Manual control for testing
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }
    
    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Reward"))
        {
            AddReward(1.0f);
            itemsCollected++;
            other.gameObject.SetActive(false);
            
            // Spawn new reward item
            if (envManager != null)
            {
                envManager.SpawnRewardItem();
            }
        }
        else if (other.CompareTag("Penalty"))
        {
            AddReward(-1.0f);
            other.gameObject.SetActive(false);
            
            if (floorMeshRenderer != null)
            {
                floorMeshRenderer.material = loseMaterial;
            }
            
            // Spawn new penalty item
            if (envManager != null)
            {
                envManager.SpawnPenaltyItem();
            }
        }
        else if (other.CompareTag("Wall"))
        {
            AddReward(-0.1f); // Small penalty for hitting walls
        }
    }
    
    private float GetDistanceToNearestItem(string itemType)
    {
        GameObject[] items = GameObject.FindGameObjectsWithTag(itemType);
        float minDistance = float.MaxValue;
        
        foreach (GameObject item in items)
        {
            if (item.activeInHierarchy)
            {
                float distance = Vector3.Distance(transform.position, item.transform.position);
                if (distance < minDistance)
                {
                    minDistance = distance;
                }
            }
        }
        
        return minDistance == float.MaxValue ? 0f : minDistance;
    }
    
    private void Update()
    {
        // Reset environment on Space key
        if (Input.GetKeyDown(KeyCode.Space))
        {
            EndEpisode();
        }
    }
}
