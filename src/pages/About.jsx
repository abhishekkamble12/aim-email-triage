import { motion } from 'framer-motion'
import { Users, Target, Lightbulb, Award } from 'lucide-react'

const About = () => {
  const team = [
    {
      name: "AI Researcher",
      role: "Reinforcement Learning Expert",
      description: "Specializes in developing adaptive AI systems for real-world applications."
    },
    {
      name: "ML Engineer",
      role: "Large Language Model Integration",
      description: "Expert in combining LLMs with traditional ML approaches for robust solutions."
    },
    {
      name: "Full-Stack Developer",
      role: "Platform Architecture",
      description: "Builds scalable web applications and APIs for AI-powered products."
    },
    {
      name: "Product Designer",
      role: "User Experience Design",
      description: "Creates intuitive interfaces for complex AI systems and technical products."
    }
  ]

  const achievements = [
    {
      icon: Target,
      title: "Novel Approach",
      description: "First system to combine RL and LLM for email triage in a POMDP framework"
    },
    {
      icon: Lightbulb,
      title: "Innovation",
      description: "Ground-breaking use of reinforcement learning for email management automation"
    },
    {
      icon: Award,
      title: "Technical Excellence",
      description: "Production-ready implementation with comprehensive evaluation metrics"
    },
    {
      icon: Users,
      title: "Real-World Impact",
      description: "Addresses critical productivity and security challenges in email management"
    }
  ]

  return (
    <div className="min-h-screen py-8 px-6">
      <div className="max-w-7xl mx-auto">
        <motion.div
          className="text-center mb-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h1 className="text-4xl font-bold mb-4">About AIM-Env</h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            A cutting-edge research project that demonstrates the power of combining reinforcement learning
            with large language models to solve complex real-world problems.
          </p>
        </motion.div>

        {/* Mission */}
        <motion.div
          className="bg-dark-200 p-8 rounded-lg mb-12 text-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <h2 className="text-3xl font-semibold mb-4">Our Mission</h2>
          <p className="text-gray-300 text-lg max-w-4xl mx-auto">
            To advance the field of AI by developing innovative solutions that combine multiple AI paradigms.
            AIM-Env serves as a proof-of-concept for how reinforcement learning and large language models
            can work together to create more intelligent, adaptive, and safe AI systems.
          </p>
        </motion.div>

        {/* Achievements */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {achievements.map((achievement, index) => (
            <motion.div
              key={achievement.title}
              className="bg-dark-200 p-6 rounded-lg text-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.05 }}
            >
              <achievement.icon className="w-12 h-12 mx-auto mb-4 text-primary-500" />
              <h3 className="text-lg font-semibold mb-2">{achievement.title}</h3>
              <p className="text-gray-300 text-sm">{achievement.description}</p>
            </motion.div>
          ))}
        </div>

        {/* Team */}
        <motion.div
          className="mb-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
        >
          <h2 className="text-3xl font-semibold text-center mb-8">The Team</h2>
          <div className="grid md:grid-cols-2 gap-8">
            {team.map((member, index) => (
              <motion.div
                key={member.name}
                className="bg-dark-200 p-6 rounded-lg"
                initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.8 + index * 0.1 }}
              >
                <h3 className="text-xl font-semibold mb-2">{member.name}</h3>
                <p className="text-primary-500 mb-3">{member.role}</p>
                <p className="text-gray-300">{member.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Technical Background */}
        <motion.div
          className="bg-dark-200 p-8 rounded-lg"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0 }}
        >
          <h2 className="text-3xl font-semibold mb-6 text-center">Technical Background</h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-xl font-semibold mb-4">Research Context</h3>
              <p className="text-gray-300 mb-4">
                This project was developed as part of an advanced AI research initiative, exploring the intersection
                of reinforcement learning and natural language processing. The system demonstrates how these
                complementary approaches can be integrated to solve complex decision-making problems.
              </p>
              <p className="text-gray-300">
                The implementation follows best practices in machine learning engineering, including modular
                design, comprehensive testing, and production-ready code quality.
              </p>
            </div>
            <div>
              <h3 className="text-xl font-semibold mb-4">Future Applications</h3>
              <ul className="space-y-2 text-gray-300">
                <li>• Intelligent document processing and routing</li>
                <li>• Automated customer service triage</li>
                <li>• Security threat detection and response</li>
                <li>• Content moderation and classification</li>
                <li>• Resource allocation in complex systems</li>
              </ul>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default About