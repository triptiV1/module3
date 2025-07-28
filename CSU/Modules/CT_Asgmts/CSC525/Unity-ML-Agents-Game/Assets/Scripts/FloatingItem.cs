using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FloatingItem : MonoBehaviour
{
    [Header("Floating Animation")]
    public float floatAmplitude = 0.2f;
    public float floatFrequency = 1f;
    public float rotationSpeed = 30f;
    
    [Header("Pulse Effect")]
    public bool enablePulse = true;
    public float pulseScale = 0.1f;
    public float pulseSpeed = 2f;
    
    private Vector3 startPosition;
    private Vector3 originalScale;
    private float timeOffset;
    
    void Start()
    {
        startPosition = transform.position;
        originalScale = transform.localScale;
        
        // Random time offset for variation
        timeOffset = Random.Range(0f, 2f * Mathf.PI);
    }
    
    void Update()
    {
        // Floating motion
        float newY = startPosition.y + Mathf.Sin((Time.time + timeOffset) * floatFrequency) * floatAmplitude;
        transform.position = new Vector3(transform.position.x, newY, transform.position.z);
        
        // Rotation
        transform.Rotate(Vector3.up, rotationSpeed * Time.deltaTime);
        
        // Pulse effect
        if (enablePulse)
        {
            float pulseValue = 1f + Mathf.Sin((Time.time + timeOffset) * pulseSpeed) * pulseScale;
            transform.localScale = originalScale * pulseValue;
        }
    }
    
    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Player") || other.GetComponent<CollectorAgent>() != null)
        {
            // Create collection effect
            CreateCollectionEffect();
        }
    }
    
    void CreateCollectionEffect()
    {
        // Simple particle effect simulation
        GameObject effect = new GameObject("CollectionEffect");
        effect.transform.position = transform.position;
        
        // Add a simple scaling effect
        StartCoroutine(CollectionEffectCoroutine(effect));
    }
    
    IEnumerator CollectionEffectCoroutine(GameObject effect)
    {
        float duration = 0.5f;
        float elapsed = 0f;
        
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float scale = Mathf.Lerp(1f, 2f, elapsed / duration);
            float alpha = Mathf.Lerp(1f, 0f, elapsed / duration);
            
            effect.transform.localScale = Vector3.one * scale;
            
            yield return null;
        }
        
        Destroy(effect);
    }
}
