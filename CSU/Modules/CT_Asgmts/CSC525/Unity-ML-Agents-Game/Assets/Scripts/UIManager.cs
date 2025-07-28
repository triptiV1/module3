using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class UIManager : MonoBehaviour
{
    [Header("UI Elements")]
    public TextMeshProUGUI scoreText;
    public TextMeshProUGUI episodeText;
    public TextMeshProUGUI timeText;
    public TextMeshProUGUI statusText;
    public Button resetButton;
    public Toggle trainingModeToggle;
    
    [Header("Agent Reference")]
    public CollectorAgent agent;
    
    private int currentEpisode = 0;
    private float episodeStartTime;
    private bool isTrainingMode = true;
    
    void Start()
    {
        episodeStartTime = Time.time;
        
        if (resetButton != null)
        {
            resetButton.onClick.AddListener(ResetEnvironment);
        }
        
        if (trainingModeToggle != null)
        {
            trainingModeToggle.onValueChanged.AddListener(ToggleTrainingMode);
            trainingModeToggle.isOn = isTrainingMode;
        }
        
        UpdateUI();
    }
    
    void Update()
    {
        UpdateUI();
    }
    
    void UpdateUI()
    {
        if (agent != null)
        {
            // Update score
            if (scoreText != null)
            {
                scoreText.text = $"Score: {agent.GetCumulativeReward():F2}";
            }
            
            // Update episode count
            if (episodeText != null)
            {
                episodeText.text = $"Episode: {currentEpisode}";
            }
            
            // Update time
            if (timeText != null)
            {
                float elapsedTime = Time.time - episodeStartTime;
                timeText.text = $"Time: {elapsedTime:F1}s";
            }
            
            // Update status
            if (statusText != null)
            {
                string mode = isTrainingMode ? "Training" : "Inference";
                statusText.text = $"Mode: {mode}";
            }
        }
    }
    
    public void OnEpisodeEnd()
    {
        currentEpisode++;
        episodeStartTime = Time.time;
    }
    
    public void ResetEnvironment()
    {
        if (agent != null)
        {
            agent.EndEpisode();
        }
        currentEpisode = 0;
        episodeStartTime = Time.time;
    }
    
    public void ToggleTrainingMode(bool trainingMode)
    {
        isTrainingMode = trainingMode;
        
        if (agent != null)
        {
            // In a real implementation, you would switch between training and inference models here
            Debug.Log($"Training mode: {(trainingMode ? "ON" : "OFF")}");
        }
    }
    
    void OnGUI()
    {
        // Simple on-screen instructions
        GUI.Box(new Rect(10, 10, 300, 120), "");
        GUI.Label(new Rect(20, 30, 280, 20), "Controls:");
        GUI.Label(new Rect(20, 50, 280, 20), "SPACE - Reset Environment");
        GUI.Label(new Rect(20, 70, 280, 20), "Arrow Keys - Manual Control (Heuristic)");
        GUI.Label(new Rect(20, 90, 280, 20), "Green Cubes = +1 Point");
        GUI.Label(new Rect(20, 110, 280, 20), "Red Cubes = -1 Point");
    }
}
