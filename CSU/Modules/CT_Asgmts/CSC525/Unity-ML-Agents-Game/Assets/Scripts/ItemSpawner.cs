using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ItemSpawner : MonoBehaviour
{
    [Header("Spawning Settings")]
    public GameObject[] itemPrefabs;
    public Transform spawnArea;
    public int maxItemsPerType = 5;
    public float spawnHeight = 0.5f;
    public float minSpawnDistance = 1f;
    
    [Header("Spawn Timing")]
    public float initialSpawnDelay = 1f;
    public float respawnDelay = 2f;
    
    private List<GameObject> spawnedItems = new List<GameObject>();
    private Bounds spawnBounds;
    
    void Start()
    {
        // Calculate spawn bounds
        if (spawnArea != null)
        {
            Renderer renderer = spawnArea.GetComponent<Renderer>();
            if (renderer != null)
            {
                spawnBounds = renderer.bounds;
            }
            else
            {
                // Default bounds if no renderer
                spawnBounds = new Bounds(spawnArea.position, Vector3.one * 10f);
            }
        }
        
        StartCoroutine(InitialSpawn());
    }
    
    IEnumerator InitialSpawn()
    {
        yield return new WaitForSeconds(initialSpawnDelay);
        SpawnInitialItems();
    }
    
    public void SpawnInitialItems()
    {
        ClearAllItems();
        
        foreach (GameObject prefab in itemPrefabs)
        {
            for (int i = 0; i < maxItemsPerType; i++)
            {
                SpawnItem(prefab);
            }
        }
    }
    
    public void SpawnItem(GameObject prefab)
    {
        Vector3 spawnPosition = GetRandomSpawnPosition();
        GameObject newItem = Instantiate(prefab, spawnPosition, Quaternion.identity, transform);
        
        // Add floating animation
        FloatingItem floatingComponent = newItem.GetComponent<FloatingItem>();
        if (floatingComponent == null)
        {
            floatingComponent = newItem.AddComponent<FloatingItem>();
        }
        
        spawnedItems.Add(newItem);
    }
    
    public void RespawnItem(GameObject prefab)
    {
        StartCoroutine(RespawnItemDelayed(prefab));
    }
    
    IEnumerator RespawnItemDelayed(GameObject prefab)
    {
        yield return new WaitForSeconds(respawnDelay);
        SpawnItem(prefab);
    }
    
    Vector3 GetRandomSpawnPosition()
    {
        Vector3 randomPosition;
        int attempts = 0;
        const int maxAttempts = 50;
        
        do
        {
            float randomX = Random.Range(spawnBounds.min.x, spawnBounds.max.x);
            float randomZ = Random.Range(spawnBounds.min.z, spawnBounds.max.z);
            randomPosition = new Vector3(randomX, spawnHeight, randomZ);
            attempts++;
        }
        while (IsPositionOccupied(randomPosition) && attempts < maxAttempts);
        
        return randomPosition;
    }
    
    bool IsPositionOccupied(Vector3 position)
    {
        Collider[] colliders = Physics.OverlapSphere(position, minSpawnDistance);
        return colliders.Length > 0;
    }
    
    public void ClearAllItems()
    {
        foreach (GameObject item in spawnedItems)
        {
            if (item != null)
            {
                DestroyImmediate(item);
            }
        }
        spawnedItems.Clear();
    }
    
    public void RemoveItem(GameObject item)
    {
        if (spawnedItems.Contains(item))
        {
            spawnedItems.Remove(item);
        }
    }
    
    void OnDrawGizmosSelected()
    {
        if (spawnArea != null)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireCube(spawnBounds.center, spawnBounds.size);
        }
    }
}
