import { motion } from 'framer-motion'

const Navbar = ({ onNavigate, currentPage }) => {
  const navItems = [
    { id: 'home', label: 'Home' },
    { id: 'demo', label: 'Live Demo' },
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'architecture', label: 'Architecture' },
    { id: 'about', label: 'About' }
  ]

  return (
    <nav className="bg-dark-200 border-b border-gray-700 px-6 py-4">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        <motion.h1
          className="text-2xl font-bold text-primary-500"
          whileHover={{ scale: 1.05 }}
        >
          AIM-Env
        </motion.h1>
        <div className="flex space-x-6">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`px-4 py-2 rounded-lg transition-colors ${
                currentPage === item.id
                  ? 'bg-primary-600 text-white'
                  : 'text-gray-300 hover:text-white hover:bg-dark-100'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>
    </nav>
  )
}

export default Navbar