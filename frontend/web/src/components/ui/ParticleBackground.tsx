import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

const ParticleBackground = () => {
    // Use state to handle hydration mismatch
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    if (!mounted) return null;

    return (
        <div className="absolute inset-0 overflow-hidden pointer-events-none z-0">
            {[...Array(30)].map((_, i) => (
                <motion.div
                    key={i}
                    className="absolute bg-blue-400 rounded-full mix-blend-screen" // slight blue tint for sci-fi feel
                    initial={{
                        x: Math.random() * window.innerWidth,
                        y: Math.random() * window.innerHeight,
                        opacity: Math.random() * 0.3 + 0.1,
                        scale: Math.random() * 0.5 + 0.5,
                    }}
                    animate={{
                        y: [0, Math.random() * window.innerHeight * (Math.random() > 0.5 ? 1 : -1)], // Move up or down
                        x: [0, Math.random() * 100 - 50], // Slight horizontal drift
                        opacity: [0.2, Math.random() * 0.5 + 0.2, 0.2],
                    }}
                    transition={{
                        duration: Math.random() * 20 + 20,
                        repeat: Infinity,
                        repeatType: "reverse",
                        ease: "linear",
                    }}
                    style={{
                        width: `${Math.random() * 3 + 1}px`,
                        height: `${Math.random() * 3 + 1}px`,
                        filter: 'blur(1px)', // Soft glow
                    }}
                />
            ))}
            {[...Array(15)].map((_, i) => (
                <motion.div
                    key={`bright-${i}`}
                    className="absolute bg-white rounded-full mix-blend-screen"
                    initial={{
                        x: Math.random() * window.innerWidth,
                        y: Math.random() * window.innerHeight,
                        opacity: Math.random() * 0.4 + 0.2,
                    }}
                    animate={{
                        opacity: [0.2, 0.8, 0.2],
                        scale: [1, 1.5, 1],
                    }}
                    transition={{
                        duration: Math.random() * 3 + 2,
                        repeat: Infinity,
                        repeatType: "reverse",
                        ease: "easeInOut",
                    }}
                    style={{
                        width: `${Math.random() * 2 + 1}px`,
                        height: `${Math.random() * 2 + 1}px`,
                        boxShadow: '0 0 4px 1px rgba(255, 255, 255, 0.3)', // Glow effect
                    }}
                />
            ))}
        </div>
    );
};

export default ParticleBackground;
