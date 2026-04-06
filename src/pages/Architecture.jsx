import { motion } from 'framer-motion'
import { Cpu, Brain, Database, ArrowRight, User, Mail, Shield } from 'lucide-react'

const Architecture = () => {
  const components = [
    {
      icon: Mail,
      title: "Email Input",
      description: "Raw email data with subject, sender, body, and metadata"
    },
    {
      icon: Brain,
      title: "LLM Agent",
      description: "Large Language Model for understanding email content and context"
    },
    {
      icon: Cpu,
      title: "RL Agent",
      description: "Reinforcement Learning agent that learns optimal triage strategies"
    },
    {
      icon: Database,
      title: "Environment State",
      description: "Maintains current email queue, user preferences, and system state"
    },
    {
      icon: Shield,
      title: "Action Execution",
      description: "Executes classification, prioritization, routing, and phishing detection"
    },
    {
      icon: User,
      title: "Reward System",
      description: "Provides feedback based on action correctness and user satisfaction"
    }
  ]

  const flow = [
    "Email arrives in inbox",
    "Agent observes current state",
    "LLM analyzes email content",
    "RL agent selects action",
    "Action executed on email",
    "Environment provides reward",
    "Agent learns from feedback"
  ]

  return (
    <div className="min-h-screen py-8 px-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8">System Architecture</h1>

        {/* Overview */}
        <motion.div
          className="bg-dark-200 p-8 rounded-lg mb-12 text-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h2 className="text-3xl font-semibold mb-4">AIM-Env: RL + LLM Email Triage</h2>
          <p className="text-gray-300 text-lg max-w-4xl mx-auto">
            A sophisticated reinforcement learning environment that combines the reasoning capabilities of large language models
            with the decision-making power of reinforcement learning to create an adaptive email management system.
          </p>
        </motion.div>

        {/* Components Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {components.map((component, index) => (
            <motion.div
              key={component.title}
              className="bg-dark-200 p-6 rounded-lg text-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.05 }}
            >
              <component.icon className="w-12 h-12 mx-auto mb-4 text-primary-500" />
              <h3 className="text-xl font-semibold mb-2">{component.title}</h3>
              <p className="text-gray-300 text-sm">{component.description}</p>
            </motion.div>
          ))}
        </div>

        {/* Data Flow */}
        <motion.div
          className="bg-dark-200 p-8 rounded-lg mb-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h2 className="text-3xl font-semibold mb-6 text-center">Data Flow & Learning Loop</h2>
          <div className="flex flex-wrap justify-center items-center gap-4">
            {flow.map((step, index) => (
              <motion.div
                key={step}
                className="flex items-center"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.2 }}
              >
                <div className="bg-primary-600 text-white px-4 py-2 rounded-lg text-sm font-medium">
                  {step}
                </div>
                {index < flow.length - 1 && (
                  <ArrowRight className="w-6 h-6 text-primary-500 mx-2" />
                )}
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Technical Details */}
        <div className="grid lg:grid-cols-2 gap-8">
          <motion.div
            className="bg-dark-200 p-6 rounded-lg"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <h3 className="text-2xl font-semibold mb-4">Reinforcement Learning Framework</h3>
            <ul className="space-y-2 text-gray-300">
              <li>• <strong>State Space:</strong> Email queue, user preferences, system status</li>
              <li>• <strong>Action Space:</strong> Classify, prioritize, route, detect phishing</li>
              <li>• <strong>Reward Function:</strong> Accuracy + efficiency + risk mitigation</li>
              <li>• <strong>Algorithm:</strong> PPO or DQN with LLM features</li>
            </ul>
          </motion.div>

          <motion.div
            className="bg-dark-200 p-6 rounded-lg"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <h3 className="text-2xl font-semibold mb-4">LLM Integration</h3>
            <ul className="space-y-2 text-gray-300">
              <li>• <strong>Content Analysis:</strong> Understand email intent and context</li>
              <li>• <strong>Feature Extraction:</strong> Generate rich embeddings</li>
              <li>• <strong>Reasoning:</strong> Explain decisions and learn from feedback</li>
              <li>• <strong>Adaptation:</strong> Handle novel email types and patterns</li>
            </ul>
          </motion.div>
        </div>

        {/* POMDP Explanation */}
        <motion.div
          className="bg-dark-200 p-8 rounded-lg mt-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h2 className="text-3xl font-semibold mb-6 text-center">Partially Observable Markov Decision Process (POMDP)</h2>
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div>
              <p className="text-gray-300 mb-4">
                Email triage is naturally a POMDP problem because agents don't have complete information about emails:
              </p>
              <ul className="space-y-2 text-gray-300">
                <li>• Hidden sender intentions</li>
                <li>• Uncertain email authenticity</li>
                <li>• Incomplete context from email content alone</li>
                <li>• Dynamic user preferences and priorities</li>
              </ul>
            </div>
            <div className="bg-dark-100 p-4 rounded-lg">
              <h4 className="font-semibold mb-2">POMDP Components in AIM-Env:</h4>
              <ul className="space-y-1 text-sm text-gray-300">
                <li><strong>States (S):</strong> Observable email features + hidden ground truth</li>
                <li><strong>Actions (A):</strong> Triage decisions (classify, prioritize, route)</li>
                <li><strong>Observations (O):</strong> Partial email information</li>
                <li><strong>Transitions (T):</strong> Email processing and queue management</li>
                <li><strong>Rewards (R):</strong> Accuracy, efficiency, security</li>
                <li><strong>Policy (π):</strong> RL + LLM decision strategy</li>
              </ul>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default Architecture