"use client";
import { Mic, StopCircle } from "lucide-react";
import { useState, useRef } from "react";

interface VoiceRecorderProps {
  onStart: () => void;
  onStop: (blob: Blob, text?: string) => void;
}

const VoiceRecorder = ({ onStart, onStop }: VoiceRecorderProps) => {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const finalTranscriptRef = useRef<string>(""); // Store final transcript

  const handleStart = async () => {
    try {
      // Start recording audio
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };
      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/wav",
        });
        onStop(audioBlob, finalTranscriptRef.current); // Pass final transcript
        audioChunksRef.current = [];
        stream.getTracks().forEach((track) => track.stop());
      };
      mediaRecorderRef.current.start();
      setIsRecording(true);
      onStart();

      // Start speech-to-text recognition
      const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) {
        console.error("SpeechRecognition API not supported in this browser.");
        return;
      }

      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = "fr-FR";

      recognitionRef.current.onresult = (event) => {
        let interimTranscript = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            // Append final transcript to the ref
            finalTranscriptRef.current += transcript + " ";
          } else {
            interimTranscript += transcript; // Ignore interim results
          }
        }
        console.log("Interim Transcript (not shown):", interimTranscript);
      };

      recognitionRef.current.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
      };

      recognitionRef.current.start();
    } catch (error) {
      console.error("Error accessing microphone:", error);
    }
  };

  const handleStop = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
  };

  return (
    <div className="flex items-center gap-2">
      {!isRecording ? (
        <button
          onClick={handleStart}
          className="p-2 rounded-lg bg-gray-200 hover:bg-gray-300"
        >
          <Mic className="h-5 w-5" />
        </button>
      ) : (
        <button
          onClick={handleStop}
          className="p-2 rounded-lg bg-red-500 hover:bg-red-600 text-white"
        >
          <StopCircle className="h-5 w-5" />
        </button>
      )}
    </div>
  );
};

export default VoiceRecorder;
