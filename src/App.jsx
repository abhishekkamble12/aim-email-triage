import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Home from './pages/Home'
import Demo from './pages/Demo'
import Dashboard from './pages/Dashboard'
import Architecture from './pages/Architecture'
import About from './pages/About'

function App() {
  const [currentPage, setCurrentPage] = useState('home')

  const pages = {
    home: Home,
    demo: Demo,
    dashboard: Dashboard,
    architecture: Architecture,
    about: About
  }

  const CurrentPage = pages[currentPage]

  return (
    <div className="min-h-screen bg-dark-300">
      <Navbar onNavigate={setCurrentPage} currentPage={currentPage} />
      <AnimatePresence mode="wait">
        <motion.div
          key={currentPage}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
        >
          <CurrentPage onNavigate={setCurrentPage} />
        </motion.div>
      </AnimatePresence>
      <Footer />
    </div>
  )
}

export default App