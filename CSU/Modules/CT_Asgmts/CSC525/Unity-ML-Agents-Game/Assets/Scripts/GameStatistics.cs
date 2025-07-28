using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

[System.Serializable]
public class EpisodeData
{
    public int episodeNumber;
    public float episodeDuration;
    public float totalReward;
    public int itemsCollected;
    public int penaltiesHit;
    public float averageRewardPerSecond;
    public string timestamp;
}

public class GameStatistics : MonoBehaviour
{
    [Header("Statistics Settings")]
    public bool enableLogging = true;
    public bool saveToFile = true;
    public string logFileName = "training_statistics.json";
    
    [Header("Current Episode Stats")]
    public int currentEpisode = 0;
    public float currentEpisodeStartTime;
    public float totalReward = 0f;
    public int itemsCollected = 0;
    public int penaltiesHit = 0;
    
    [Header("Overall Stats")]
    public float bestEpisodeReward = float.MinValue;
    public float averageReward = 0f;
    public int totalEpisodes = 0;
    public float totalTrainingTime = 0f;
    
    private List<EpisodeData> episodeHistory = new List<EpisodeData>();
    private string logFilePath;
    
    void Start()
    {
        logFilePath = Path.Combine(Application.persistentDataPath, logFileName);
        LoadStatistics();
        StartNewEpisode();
    }
    
    public void StartNewEpisode()
    {
        currentEpisode++;
        currentEpisodeStartTime = Time.time;
        totalReward = 0f;
        itemsCollected = 0;
        penaltiesHit = 0;
        
        if (enableLogging)
        {
            Debug.Log($"Episode {currentEpisode} started");
        }
    }
    
    public void EndEpisode()
    {
        float episodeDuration = Time.time - currentEpisodeStartTime;
        float rewardPerSecond = episodeDuration > 0 ? totalReward / episodeDuration : 0f;
        
        // Create episode data
        EpisodeData episodeData = new EpisodeData
        {
            episodeNumber = currentEpisode,
            episodeDuration = episodeDuration,
            totalReward = totalReward,
            itemsCollected = itemsCollected,
            penaltiesHit = penaltiesHit,
            averageRewardPerSecond = rewardPerSecond,
            timestamp = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")
        };
        
        episodeHistory.Add(episodeData);
        
        // Update overall statistics
        UpdateOverallStats(episodeData);
        
        if (enableLogging)
        {
            LogEpisodeResults(episodeData);
        }
        
        if (saveToFile)
        {
            SaveStatistics();
        }
    }
    
    public void AddReward(float reward)
    {
        totalReward += reward;
    }
    
    public void IncrementItemsCollected()
    {
        itemsCollected++;
    }
    
    public void IncrementPenalties()
    {
        penaltiesHit++;
    }
    
    void UpdateOverallStats(EpisodeData episode)
    {
        totalEpisodes++;
        totalTrainingTime += episode.episodeDuration;
        
        if (episode.totalReward > bestEpisodeReward)
        {
            bestEpisodeReward = episode.totalReward;
        }
        
        // Calculate running average reward
        float totalRewardSum = 0f;
        foreach (var ep in episodeHistory)
        {
            totalRewardSum += ep.totalReward;
        }
        averageReward = totalRewardSum / episodeHistory.Count;
    }
    
    void LogEpisodeResults(EpisodeData episode)
    {
        Debug.Log($"Episode {episode.episodeNumber} Complete:");
        Debug.Log($"  Duration: {episode.episodeDuration:F2}s");
        Debug.Log($"  Total Reward: {episode.totalReward:F2}");
        Debug.Log($"  Items Collected: {episode.itemsCollected}");
        Debug.Log($"  Penalties Hit: {episode.penaltiesHit}");
        Debug.Log($"  Reward/Second: {episode.averageRewardPerSecond:F2}");
        Debug.Log($"  Best Episode Reward: {bestEpisodeReward:F2}");
        Debug.Log($"  Average Reward: {averageReward:F2}");
    }
    
    void SaveStatistics()
    {
        try
        {
            string json = JsonUtility.ToJson(new SerializableList<EpisodeData>(episodeHistory), true);
            File.WriteAllText(logFilePath, json);
            
            if (enableLogging)
            {
                Debug.Log($"Statistics saved to: {logFilePath}");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to save statistics: {e.Message}");
        }
    }
    
    void LoadStatistics()
    {
        try
        {
            if (File.Exists(logFilePath))
            {
                string json = File.ReadAllText(logFilePath);
                SerializableList<EpisodeData> loadedData = JsonUtility.FromJson<SerializableList<EpisodeData>>(json);
                episodeHistory = loadedData.items;
                
                // Restore overall stats
                if (episodeHistory.Count > 0)
                {
                    currentEpisode = episodeHistory[episodeHistory.Count - 1].episodeNumber;
                    UpdateOverallStatsFromHistory();
                }
                
                if (enableLogging)
                {
                    Debug.Log($"Loaded {episodeHistory.Count} episodes from statistics file");
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to load statistics: {e.Message}");
        }
    }
    
    void UpdateOverallStatsFromHistory()
    {
        totalEpisodes = episodeHistory.Count;
        bestEpisodeReward = float.MinValue;
        float totalRewardSum = 0f;
        totalTrainingTime = 0f;
        
        foreach (var episode in episodeHistory)
        {
            if (episode.totalReward > bestEpisodeReward)
            {
                bestEpisodeReward = episode.totalReward;
            }
            totalRewardSum += episode.totalReward;
            totalTrainingTime += episode.episodeDuration;
        }
        
        averageReward = totalRewardSum / episodeHistory.Count;
    }
    
    public List<EpisodeData> GetEpisodeHistory()
    {
        return new List<EpisodeData>(episodeHistory);
    }
    
    public void ResetStatistics()
    {
        episodeHistory.Clear();
        currentEpisode = 0;
        totalEpisodes = 0;
        bestEpisodeReward = float.MinValue;
        averageReward = 0f;
        totalTrainingTime = 0f;
        
        if (saveToFile && File.Exists(logFilePath))
        {
            File.Delete(logFilePath);
        }
        
        Debug.Log("Statistics reset");
    }
}

[System.Serializable]
public class SerializableList<T>
{
    public List<T> items;
    
    public SerializableList(List<T> items)
    {
        this.items = items;
    }
}
