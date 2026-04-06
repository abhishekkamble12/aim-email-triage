import { motion } from 'framer-motion'
import { Github, Mail, ExternalLink } from 'lucide-react'

const Footer = () => {
  return (
    <footer className="bg-dark-200 border-t border-gray-700 py-8 px-6 mt-16">
      <div className="max-w-7xl mx-auto">
        <div className="grid md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-xl font-semibold text-primary-500 mb-4">AIM-Env</h3>
            <p className="text-gray-300 text-sm">
              Adaptive Intelligent Mail Environment - Revolutionizing email triage with AI.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-4">Quick Links</h4>
            <ul className="space-y-2 text-sm text-gray-300">
              <li><a href="#" className="hover:text-primary-500 transition-colors">Documentation</a></li>
              <li><a href="#" className="hover:text-primary-500 transition-colors">API Reference</a></li>
              <li><a href="#" className="hover:text-primary-500 transition-colors">Research Paper</a></li>
              <li><a href="#" className="hover:text-primary-500 transition-colors">GitHub</a></li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-4">Contact</h4>
            <div className="flex space-x-4">
              <motion.a
                href="#"
                className="text-gray-300 hover:text-primary-500 transition-colors"
                whileHover={{ scale: 1.1 }}
              >
                <Github className="w-5 h-5" />
              </motion.a>
              <motion.a
                href="#"
                className="text-gray-300 hover:text-primary-500 transition-colors"
                whileHover={{ scale: 1.1 }}
              >
                <Mail className="w-5 h-5" />
              </motion.a>
              <motion.a
                href="#"
                className="text-gray-300 hover:text-primary-500 transition-colors"
                whileHover={{ scale: 1.1 }}
              >
                <ExternalLink className="w-5 h-5" />
              </motion.a>
            </div>
          </div>
        </div>
        <div className="border-t border-gray-700 mt-8 pt-8 text-center text-sm text-gray-400">
          <p>&copy; 2026 AIM-Env Team. Built with React, FastAPI, and ❤️ for AI innovation.</p>
        </div>
      </div>
    </footer>
  )
}

export default Footer