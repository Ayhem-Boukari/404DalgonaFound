/* eslint-disable @next/next/no-img-element */
"use client";
import { useState, useRef, useEffect } from "react";
import {
  Sun,
  Moon,
  Send,
  Loader2,
  ThumbsUp,
  ThumbsDown,
  Copy,
} from "lucide-react";
import Lottie from "lottie-react";
import botAnimation from "../../public/bot.json";
import VoiceRecorder from "./VoiceRecorder";
import CustomAudioPlayer from "./CustomAudioPlayer";
import { categories } from "@/lib/constants";
import toast from "react-hot-toast";

interface Message {
  text?: string;
  sender: "user" | "bot";
  audioUrl?: string;
}

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>("");
  const [isPending, setIsPending] = useState<boolean>(false);
  const [isDarkMode, setIsDarkMode] = useState<boolean>(false);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendQueryToBackend = async (query: string) => {
    try {
      const response = await fetch("http://127.0.0.1:8000/query/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }), // Send query in the expected format
      });

      if (!response.ok) {
        throw new Error("Failed to fetch response from the backend");
      }

      const data = await response.json();
      console.log(data);
      return data.answer; // Assuming the backend returns { "response": "..." }
    } catch (error) {
      console.error("Error sending query to backend:", error);
      return "Désolé, quelque chose s'est mal passé. Veuillez réessayer.";
    }
  };

  const handleSendMessage = async () => {
    if (input.trim() === "") return;

    // Add user's message to the chat
    const userMessage: Message = { text: input, sender: "user" };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsPending(true);

    try {
      // Send the query to the backend and get the bot's response
      const botResponse = await sendQueryToBackend(input);
      const botMessage: Message = { text: botResponse, sender: "bot" };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error(error);
      const errorMessage: Message = {
        text: "Désolé, quelque chose s'est mal passé. Veuillez réessayer.",
        sender: "bot",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsPending(false);
    }
  };

  const handleQuickReply = async (text: string) => {
    // Add quick reply as a user message
    const userMessage: Message = { text, sender: "user" };
    setMessages((prev) => [...prev, userMessage]);
    setIsPending(true);

    try {
      // Send the quick reply text to the backend and get the bot's response
      const botResponse = await sendQueryToBackend(text);
      const botMessage: Message = { text: botResponse, sender: "bot" };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error(error);
      const errorMessage: Message = {
        text: "Désolé, quelque chose s'est mal passé. Veuillez réessayer.",
        sender: "bot",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsPending(false);
    }
  };

  const handleVoiceMessage = (audioBlob: Blob, text?: string) => {
    const audioUrl = URL.createObjectURL(audioBlob);
    const userMessage: Message = { sender: "user", audioUrl, text }; // Include text in the message
    setMessages((prev) => [...prev, userMessage]);
    setIsRecording(false); // Reset recording state

    // If text is available, send it to the backend
    if (text) {
      console.log("text", text);
      setIsPending(true);
      sendQueryToBackend(text)
        .then((response) => {
          const botMessage: Message = { text: response, sender: "bot" };
          setMessages((prev) => [...prev, botMessage]);
        })
        .catch((error) => {
          console.error(error);
          const errorMessage: Message = {
            text: "Désolé, quelque chose s'est mal passé. Veuillez réessayer.",
            sender: "bot",
          };
          setMessages((prev) => [...prev, errorMessage]);
        })
        .finally(() => {
          setIsPending(false);
        });
    } else {
      // If no text is available, show an error message
      const errorMessage: Message = {
        text: "Désolé, je n'ai pas pu traiter votre message vocal. Veuillez réessayer.",
        sender: "bot",
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  return (
    <div className="fixed bottom-8 sm:right-8 right-1 flex flex-col">
      {/* Attractive Message */}
      {!isOpen && (
        <div
          className={`absolute right-20 bottom-4 p-2 rounded-lg shadow-md border-secondary flex items-center gap-2 w-fit ${
            isDarkMode ? "bg-gray-800 text-white" : "bg-white text-gray-800"
          }`}
        >
          <span className="text-sm text-end truncate select-none">
            Besoin d'aide ? Discutez avec moi !
          </span>
          <div className="w-4 h-4 bg-green-500 rounded-full animate-pulse" />
        </div>
      )}

      {/* Lottie Button */}
      <div className="p-[0.2rem] size-fit border border-secondary rounded-full flex items-center justify-center shadow-md place-self-end">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="p-0 bg-secondary rounded-full"
        >
          <div className="size-16">
            <Lottie animationData={botAnimation} loop={true} />
          </div>
        </button>
      </div>

      {/* Chatbot Window */}
      {isOpen && (
        <div
          className={`mt-4 ml-1 max-h-[80vh] max-w-[640px] rounded-2xl shadow-2xl flex flex-col overflow-hidden ${
            isDarkMode ? "bg-gray-900" : "bg-background-white"
          }`}
        >
          {/* Chatbot Header */}
          <div
            className={`p-4 ${
              isDarkMode
                ? "bg-gray-800"
                : "bg-gradient-to-br from-primary to-primary/90"
            } rounded-t-xl`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <img
                  src={"/logos/icon-white.svg"}
                  alt="University Logo"
                  className="h-8 w-auto"
                />
                <h2
                  className={`text-lg font-semibold font-canela ${
                    isDarkMode ? "text-white" : "text-text-white"
                  }`}
                >
                  Chatbot IHEC
                </h2>
              </div>
              <button
                onClick={toggleDarkMode}
                className="p-2 rounded-full hover:bg-opacity-20 hover:bg-white transition-colors duration-200"
              >
                {isDarkMode ? (
                  <Moon className="h-5 w-5 text-white" />
                ) : (
                  <Sun className="h-5 w-5 text-white" />
                )}
              </button>
            </div>
            <p
              className={`text-sm ${
                isDarkMode ? "text-gray-300" : "text-text-white/80"
              }`}
            >
              Comment puis-je vous aider aujourd'hui ?
            </p>
          </div>

          {/* Chat Messages */}
          <div
            className={`flex-1 p-4 overflow-y-auto ${
              isDarkMode ? "bg-gray-800" : "bg-background-light"
            }`}
            style={{ maxHeight: "calc(80vh - 200px)" }}
          >
            {messages.length === 0 && (
              <div className="my-2 flex justify-start">
                <div
                  className={`max-w-[70%] p-3 rounded-lg ${
                    isDarkMode ? "bg-gray-700 text-white" : "bg-gray-100"
                  }`}
                >
                  Bonjour ! Je suis là pour vous aider avec vos questions sur
                  l'IHEC. Comment puis-je vous aider aujourd'hui ?
                </div>
              </div>
            )}

            {messages.map((msg, index) => (
              <div
                key={index}
                className={`my-2 flex ${
                  msg.sender === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {msg.sender === "bot" && (
                  <div className="mr-2 mt-2 flex items-start">
                    <div
                      className={`h-8 w-8 rounded-full flex items-center justify-center ${
                        isDarkMode ? "bg-gray-700" : "bg-gray-200"
                      }`}
                    >
                      <img
                        src="/bot-avatar.png"
                        alt="Bot Avatar"
                        className="h-8 w-8 rounded-full"
                      />
                    </div>
                  </div>
                )}
                <div
                  className={`max-w-[70%] p-3 rounded-lg ${
                    msg.sender === "user"
                      ? isDarkMode
                        ? "bg-primary text-white"
                        : "bg-primary text-text-white"
                      : isDarkMode
                      ? "bg-gray-700 text-white"
                      : "bg-gray-100 text-text-primary"
                  }`}
                >
                  {msg.text}
                  {msg.audioUrl && <CustomAudioPlayer src={msg.audioUrl} />}

                  {/* Add Like, Dislike, and Copy buttons for bot messages */}
                  {msg.sender === "bot" && (
                    <div className="flex items-center gap-2 mt-2">
                      <button
                        onClick={() => console.log("Liked:", msg.text)}
                        className={`p-1 rounded-md hover:bg-opacity-20 ${
                          isDarkMode ? "hover:bg-white" : "hover:bg-gray-200"
                        } transition-colors duration-200`}
                      >
                        <ThumbsUp className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => console.log("Disliked:", msg.text)}
                        className={`p-1 rounded-md hover:bg-opacity-20 ${
                          isDarkMode ? "hover:bg-white" : "hover:bg-gray-200"
                        } transition-colors duration-200`}
                      >
                        <ThumbsDown className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(msg.text || "");
                          toast.success("Réponse copiée !", {
                            position: "bottom-right",
                            duration: 2000,
                          });

                          console.log("Copied:", msg.text);
                        }}
                        className={`p-1 rounded-md hover:bg-opacity-20 ${
                          isDarkMode ? "hover:bg-white" : "hover:bg-gray-200"
                        } transition-colors duration-200`}
                      >
                        <Copy className="h-4 w-4" />
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {isPending && (
              <div className="my-2 flex justify-start">
                <div
                  className={`max-w-[70%] p-3 rounded-lg ${
                    isDarkMode ? "bg-gray-700 text-white" : "bg-gray-100"
                  } flex items-center gap-2`}
                >
                  <Loader2 className="h-4 w-4 animate-spin" />
                </div>
              </div>
            )}

            {messages.length === 0 && (
              <div className="flex flex-wrap gap-2 mt-4">
                {categories.slice(0, 5).map((category, index) => (
                  <button
                    key={index}
                    onClick={() => handleQuickReply(category)}
                    className={`px-3 py-2 rounded-lg text-sm text-start w-fit ${
                      isDarkMode
                        ? "bg-gray-700 text-white hover:bg-gray-600"
                        : "bg-gray-100 hover:bg-gray-200"
                    }`}
                  >
                    {category}
                  </button>
                ))}
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div
            className={`p-4 border-t ${
              isDarkMode ? "border-gray-700 bg-gray-800" : "border-gray-200"
            }`}
          >
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
                className={`flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 ${
                  isDarkMode
                    ? "bg-gray-700 border-gray-600 text-white focus:ring-primary"
                    : "border-gray-300 focus:ring-primary"
                }`}
                placeholder="Tapez un message..."
                disabled={isPending || isRecording}
              />
              <VoiceRecorder
                onStart={() => setIsRecording(true)}
                onStop={handleVoiceMessage}
              />
              <button
                onClick={handleSendMessage}
                className={`p-2 rounded-lg ${
                  isDarkMode
                    ? "bg-primary text-white hover:bg-primary/90"
                    : "bg-primary text-text-white hover:bg-primary/90"
                } transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed`}
                disabled={isPending || isRecording}
              >
                <Send className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Chatbot;