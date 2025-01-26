/* eslint-disable @typescript-eslint/no-explicit-any */
// utils/speechToText.ts
export const startSpeechToText = (
  onResult: (text: string) => void,
  onError: (error: string) => void
) => {
  const SpeechRecognition =
    (window as any).SpeechRecognition ||
    (window as any).webkitSpeechRecognition;

  if (!SpeechRecognition) {
    onError("Speech recognition is not supported in this browser.");
    return null;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = "en-US"; // Set the language
  recognition.interimResults = false; // Only final results
  recognition.maxAlternatives = 1; // Only one result

  recognition.onresult = (event: any) => {
    console.log("SpeechRecognition result:", event); // Debug log
    const transcript = event.results[0][0].transcript;
    onResult(transcript);
  };

  recognition.onerror = (event: any) => {
    console.error("SpeechRecognition error:", event.error); // Debug log
    onError(event.error);
  };

  recognition.onstart = () => {
    console.log("SpeechRecognition started"); // Debug log
  };

  recognition.onend = () => {
    console.log("SpeechRecognition ended"); // Debug log
  };

  recognition.start();
  return recognition;
};
