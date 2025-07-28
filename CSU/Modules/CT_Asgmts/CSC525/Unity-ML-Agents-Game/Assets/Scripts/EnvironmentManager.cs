using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnvironmentManager : MonoBehaviour
{
    [Header("Item Prefabs")]
    public GameObject rewardItemPrefab;
    public GameObject penaltyItemPrefab;
    
    [Header("Spawn Settings")]
    public int initialRewardItems = 5;
    public int initialPenaltyItems = 3;
    public float spawnAreaRadius = 8f;
    public float minDistanceFromAgent = 2f;
    
    [Header("Environment Bounds")]
    public Transform platform;
    public Transform agent;
    
    private List<GameObject> activeRewardItems = new List<GameObject>();
    private List<GameObject> activePenaltyItems = new List<GameObject>();
    
    void Start()
    {
        ResetEnvironment();
    }
    
    public void ResetEnvironment()
    {
        // Clear existing items
        ClearAllItems();
        
        // Spawn initial items
        for (int i = 0; i < initialRewardItems; i++)
        {
            SpawnRewardItem();
        }
        
        for (int i = 0; i < initialPenaltyItems; i++)
        {
            SpawnPenaltyItem();
        }
    }
    
    public void SpawnRewardItem()
    {
        Vector3 spawnPosition = GetRandomSpawnPosition();
        GameObject newItem = Instantiate(rewardItemPrefab, spawnPosition, Quaternion.identity, transform);
        newItem.tag = "Reward";
        activeRewardItems.Add(newItem);
    }
    
    public void SpawnPenaltyItem()
    {
        Vector3 spawnPosition = GetRandomSpawnPosition();
        GameObject newItem = Instantiate(penaltyItemPrefab, spawnPosition, Quaternion.identity, transform);
        newItem.tag = "Penalty";
        activePenaltyItems.Add(newItem);
    }
    
    private Vector3 GetRandomSpawnPosition()
    {
        Vector3 spawnPosition;
        int attempts = 0;
        const int maxAttempts = 50;
        
        do
        {
            // Generate random position within spawn area
            float randomX = Random.Range(-spawnAreaRadius, spawnAreaRadius);
            float randomZ = Random.Range(-spawnAreaRadius, spawnAreaRadius);
            spawnPosition = transform.position + new Vector3(randomX, 0.5f, randomZ);
            attempts++;
        }
        while (Vector3.Distance(spawnPosition, agent.position) < minDistanceFromAgent && attempts < maxAttempts);
        
        return spawnPosition;
    }
    
    private void ClearAllItems()
    {
        // Clear reward items
        foreach (GameObject item in activeRewardItems)
        {
            if (item != null)
            {
                DestroyImmediate(item);
            }
        }
        activeRewardItems.Clear();
        
        // Clear penalty items
        foreach (GameObject item in activePenaltyItems)
        {
            if (item != null)
            {
                DestroyImmediate(item);
            }
        }
        activePenaltyItems.Clear();
    }
    
    void OnDrawGizmosSelected()
    {
        // Draw spawn area
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(transform.position, spawnAreaRadius);
        
        // Draw minimum distance from agent
        if (agent != null)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawWireSphere(agent.position, minDistanceFromAgent);
        }
    }
}
