import React, { useRef, useState } from "react";
import { Play, Pause } from "lucide-react";

interface CustomAudioPlayerProps {
  src: string;
}

const CustomAudioPlayer: React.FC<CustomAudioPlayerProps> = ({ src }) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const togglePlayPause = () => {
    if (!audioRef.current) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
  };

  const handlePlay = () => setIsPlaying(true);

  const handlePause = () => setIsPlaying(false);

  const handleEnd = () => setIsPlaying(false);

  return (
    <div className="flex items-center gap-2">
      <button onClick={togglePlayPause}>
        {isPlaying ? (
          <Pause className="h-5 w-5" />
        ) : (
          <Play className="h-5 w-5" />
        )}
      </button>
      <audio
        ref={audioRef}
        src={src}
        onPlay={handlePlay}
        onPause={handlePause}
        onEnded={handleEnd}
        className="hidden"
      />
    </div>
  );
};

export default CustomAudioPlayer;
