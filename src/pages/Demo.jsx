import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Play, Pause, RotateCcw, Mail, AlertTriangle, CheckCircle, Loader } from 'lucide-react'
import api from '../services/api'

const Demo = () => {
  const [isRunning, setIsRunning] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [demoData, setDemoData] = useState(null)
  const [selectedEmail, setSelectedEmail] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  const startDemo = async () => {
    setIsRunning(true)
    setCurrentStep(0)
    setIsLoading(true)
    setError(null)
    try {
      const response = await api.post('/run-demo', {
        difficulty: 'easy',
        agent_type: 'heuristic'
      })
      setDemoData(response.data)
    } catch (error) {
      console.error('Demo failed:', error)
      setError('Failed to start demo. Please try again.')
      setIsRunning(false)
    } finally {
      setIsLoading(false)
    }
  }

  const nextStep = () => {
    if (demoData && currentStep < demoData.steps.length - 1) {
      setCurrentStep(currentStep + 1)
    } else {
      setIsRunning(false)
    }
  }

  useEffect(() => {
    if (isRunning && demoData) {
      const timer = setTimeout(nextStep, 2000)
      return () => clearTimeout(timer)
    }
  }, [isRunning, currentStep, demoData])

  const currentStepData = demoData?.steps[currentStep]
  const currentEmail = demoData?.emails.find(email =>
    currentStepData?.observation.pending_emails?.some(pe => pe.id === email.id)
  )

  return (
    <div className="min-h-screen py-8 px-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8">Live Email Triage Demo</h1>

        {/* Controls */}
        <div className="flex justify-center gap-4 mb-8">
          <button
            onClick={startDemo}
            disabled={isRunning || isLoading}
            className="bg-primary-600 hover:bg-primary-700 disabled:bg-gray-600 px-6 py-3 rounded-lg flex items-center gap-2"
          >
            {isLoading ? <Loader className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5" />}
            {isLoading ? 'Starting...' : 'Start Demo'}
          </button>
          <button
            onClick={() => setIsRunning(!isRunning)}
            disabled={!demoData || isLoading}
            className="bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-600 px-6 py-3 rounded-lg flex items-center gap-2"
          >
            {isRunning ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
            {isRunning ? 'Pause' : 'Resume'}
          </button>
          <button
            onClick={() => {
              setDemoData(null)
              setCurrentStep(0)
              setIsRunning(false)
              setError(null)
            }}
            className="bg-gray-600 hover:bg-gray-700 px-6 py-3 rounded-lg flex items-center gap-2"
          >
            <RotateCcw className="w-5 h-5" />
            Reset
          </button>
        </div>

        {error && (
          <div className="bg-red-900/20 border border-red-500 text-red-400 px-4 py-3 rounded-lg mb-8 text-center">
            {error}
          </div>
        )}

        {demoData && (
          <div className="grid lg:grid-cols-2 gap-8">
            {/* Email Inbox */}
            <div className="bg-dark-200 rounded-lg p-6">
              <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                <Mail className="w-6 h-6" />
                Email Inbox
              </h2>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {demoData.emails.map((email, index) => (
                  <motion.div
                    key={email.id}
                    className={`p-4 rounded-lg border cursor-pointer transition-colors ${
                      selectedEmail?.id === email.id
                        ? 'border-primary-500 bg-primary-500/10'
                        : 'border-gray-600 hover:border-gray-500'
                    }`}
                    onClick={() => setSelectedEmail(email)}
                    whileHover={{ scale: 1.02 }}
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-semibold">{email.subject}</h3>
                        <p className="text-sm text-gray-400">{email.sender}</p>
                      </div>
                      {email.is_phishing && (
                        <AlertTriangle className="w-5 h-5 text-red-500" />
                      )}
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Agent Actions & Rewards */}
            <div className="bg-dark-200 rounded-lg p-6">
              <h2 className="text-2xl font-semibold mb-4">Agent Decision Process</h2>

              {currentStepData && (
                <motion.div
                  className="space-y-4"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  key={currentStep}
                >
                  <div className="bg-dark-100 p-4 rounded-lg">
                    <h3 className="font-semibold mb-2">Step {currentStep + 1}</h3>
                    <p className="text-sm text-gray-300 mb-2">Action: {currentStepData.action}</p>
                    <div className="flex items-center gap-2">
                      <span className="text-sm">Reward:</span>
                      <span className={`font-semibold ${
                        currentStepData.reward > 0 ? 'text-green-500' :
                        currentStepData.reward < 0 ? 'text-red-500' : 'text-yellow-500'
                      }`}>
                        {currentStepData.reward.toFixed(2)}
                      </span>
                    </div>
                  </div>

                  {currentEmail && (
                    <div className="bg-dark-100 p-4 rounded-lg">
                      <h3 className="font-semibold mb-2">Current Email Analysis</h3>
                      <p className="text-sm"><strong>Subject:</strong> {currentEmail.subject}</p>
                      <p className="text-sm"><strong>Sender:</strong> {currentEmail.sender}</p>
                      {currentEmail.category && (
                        <p className="text-sm"><strong>Category:</strong> {currentEmail.category}</p>
                      )}
                      {currentEmail.priority && (
                        <p className="text-sm"><strong>Priority:</strong> {currentEmail.priority}</p>
                      )}
                      {currentEmail.is_phishing !== null && (
                        <p className="text-sm flex items-center gap-1">
                          <strong>Phishing:</strong>
                          {currentEmail.is_phishing ?
                            <AlertTriangle className="w-4 h-4 text-red-500" /> :
                            <CheckCircle className="w-4 h-4 text-green-500" />
                          }
                        </p>
                      )}
                    </div>
                  )}

                  <div className="bg-dark-100 p-4 rounded-lg">
                    <h3 className="font-semibold mb-2">Final Score</h3>
                    <p className="text-2xl font-bold text-primary-500">
                      {demoData.final_score.toFixed(3)}
                    </p>
                  </div>
                </motion.div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default Demo