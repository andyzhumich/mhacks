import React, { useEffect, useState } from 'react';

const IndexPage: React.FC = () => {
  const [confidence, setConfidence] = useState<number>(0.0);
  const [label, setLabel] = useState<string>("False");
  const [recording, setRecording] = useState<boolean>(false);
  
  useEffect(() => {
    const eventSource = new EventSource('http://localhost:8080/audio_stream');
    eventSource.onmessage = (event) => {
        const [newConfidence, newLabel] = event.data.split(',');
        setConfidence(parseFloat(newConfidence));
        setLabel(newLabel);
    };

    return () => {
        eventSource.close();
    };
  }, []);


  const toggleRecording = async () => {
    try {
      if (recording) {
        // If already recording, stop recording
        setRecording(false);
      } else {
        // If not recording, start recording
        const response = await fetch('http://localhost:8080/start_recording', { method: 'POST' });
        const data = await response.json();
        console.log(data);
        setRecording(true);
      }
    } catch (error) {
      console.error('Error toggling recording:', error);
    }
  };

  return (
    <div>
      <h1>Audio Classification Demo</h1>
      <div>
        <p>Confidence: {confidence.toFixed(2)}</p>
        <p>Danger: {label}</p>
        <progress max={1} value={confidence}></progress>
      </div>
      <button onClick={toggleRecording}>
        {recording ? 'Stop Recording' : 'Start Recording'}
      </button>
    </div>
  );
};

export default IndexPage;
