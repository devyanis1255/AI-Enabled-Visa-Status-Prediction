import { useState, useEffect } from 'react'
import { motion, AnimatePresence, useMotionValue, useTransform, animate } from 'framer-motion'
import { Rocket, Loader2, Sparkles, X, Sun, Moon, Cpu, BarChart3, Activity, Database } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import './App.css'

const API_BASE = 'http://localhost:8000';

const analysisData = [
  { year: '2016', days: 120 },
  { year: '2017', days: 140 },
  { year: '2018', days: 125 },
  { year: '2019', days: 110 },
  { year: '2020', days: 250 },
  { year: '2021', days: 180 },
  { year: '2022', days: 160 },
  { year: '2023', days: 145 },
]

function AnimatedCounter({ value }) {
  const count = useMotionValue(0)
  const rounded = useTransform(count, Math.round)

  useEffect(() => {
    const animation = animate(count, value, { 
      duration: 2, 
      ease: "easeOut" 
    })
    return animation.stop
  }, [value, count])

  return <motion.span>{rounded}</motion.span>
}

function AppModal({ isOpen, onClose, title, children, large }) {
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div 
          className="modal-overlay"
          initial={{ opacity: 0, backdropFilter: 'blur(0px)' }}
          animate={{ opacity: 1, backdropFilter: 'blur(10px)' }}
          exit={{ opacity: 0, backdropFilter: 'blur(0px)' }}
          onClick={onClose}
        >
          <motion.div 
            className="result-modal glass-card"
            initial={{ scale: 0.8, y: 50, opacity: 0 }}
            animate={{ scale: 1, y: 0, opacity: 1 }}
            exit={{ scale: 0.8, y: 50, opacity: 0 }}
            transition={{ type: 'spring', damping: 20, stiffness: 100 }}
            onClick={e => e.stopPropagation()}
            style={{ 
              textAlign: 'left', 
              padding: '3rem',
              maxWidth: large ? '850px' : '500px',
              maxHeight: '90vh',
              overflowY: 'auto'
            }}
          >
            <button className="close-btn" type="button" onClick={onClose}><X size={24} /></button>
            <h2 className="modal-title">{title}</h2>
            <div className="modal-body">
               {children}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

function MetricCard({ icon, title, value, delay, onClick }) {
  return (
    <motion.div 
      className="metric-card"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, type: 'spring', stiffness: 100 }}
      whileHover={{ y: -5 }}
      onClick={onClick}
    >
      <div className="metric-icon-wrapper">
        {icon}
      </div>
      <div className="metric-content">
        <h4>{title}</h4>
        <h2>{value}</h2>
      </div>
    </motion.div>
  )
}

function App() {
  const [theme, setTheme] = useState('light')
  const [options, setOptions] = useState(null)
  const [loadingOptions, setLoadingOptions] = useState(true)
  const [activeModal, setActiveModal] = useState(null)
  
  const [formData, setFormData] = useState({
    case_status: '',
    visa_class: '',
    pw_wage_level: '',
    employer_city: '',
    worksite_city: '',
    app_date: new Date().toISOString().split('T')[0],
    dec_date: new Date().toISOString().split('T')[0],
    app_year: new Date().getFullYear(),
    app_month: new Date().getMonth() + 1,
    dec_year: new Date().getFullYear(),
    dec_month: new Date().getMonth() + 1
  })

  const [prediction, setPrediction] = useState(null)
  const [isPredicting, setIsPredicting] = useState(false)

  // Initialize theme
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme')
    if (savedTheme) {
      setTheme(savedTheme)
      document.documentElement.setAttribute('data-theme', savedTheme)
    } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      setTheme('dark')
      document.documentElement.setAttribute('data-theme', 'dark')
    }
  }, [])

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    setTheme(newTheme)
    document.documentElement.setAttribute('data-theme', newTheme)
    localStorage.setItem('theme', newTheme)
  }

  useEffect(() => {
    fetch(`${API_BASE}/options`)
      .then(res => res.json())
      .then(data => {
        setOptions(data)
        if (data.case_status?.length > 0) {
          setFormData(prev => ({
            ...prev,
            case_status: data.case_status[0],
            visa_class: data.visa_class?.[0] || '',
            pw_wage_level: data.pw_wage_level?.[0] || '',
            employer_city: data.employer_city?.[0] || '',
            worksite_city: data.worksite_city?.[0] || ''
          }))
        }
        setLoadingOptions(false)
      })
      .catch(err => {
        console.error("Failed to load options", err)
        setLoadingOptions(false)
      })
  }, [])

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: isNaN(value) ? value : Number(value) }))
  }

  const handleDateChange = (e) => {
    const { name, value } = e.target;
    if (value) {
      const date = new Date(value);
      if (name === 'app_date') {
        setFormData(prev => ({ ...prev, app_date: value, app_year: date.getFullYear(), app_month: date.getMonth() + 1 }));
      } else if (name === 'dec_date') {
        setFormData(prev => ({ ...prev, dec_date: value, dec_year: date.getFullYear(), dec_month: date.getMonth() + 1 }));
      }
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsPredicting(true);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })
      const data = await res.json()
      if (data.estimated_days !== undefined) {
        setPrediction(data.estimated_days)
      }
    } catch (err) {
      console.error(err)
    } finally {
      setIsPredicting(false)
    }
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1, delayChildren: 0.3 }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 30, scale: 0.95 },
    show: { opacity: 1, y: 0, scale: 1, transition: { type: 'spring', stiffness: 100 } }
  }

  return (
    <>
      <motion.button 
        className="theme-toggle"
        onClick={toggleTheme}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={theme}
            initial={{ opacity: 0, rotate: -90 }}
            animate={{ opacity: 1, rotate: 0 }}
            exit={{ opacity: 0, rotate: 90 }}
            transition={{ duration: 0.2 }}
          >
            {theme === 'light' ? <Moon size={24} /> : <Sun size={24} />}
          </motion.div>
        </AnimatePresence>
      </motion.button>

      <motion.div 
        className="orb orb-1"
        animate={{ 
          x: [0, 50, -50, 0], 
          y: [0, 30, -30, 0],
          scale: [1, 1.1, 0.9, 1]
        }}
        transition={{ repeat: Infinity, duration: 15, ease: "easeInOut" }}
      />
      <motion.div 
        className="orb orb-2"
        animate={{ 
          x: [0, -60, 40, 0], 
          y: [0, -40, 50, 0],
          scale: [1, 1.2, 0.8, 1]
        }}
        transition={{ repeat: Infinity, duration: 20, ease: "easeInOut" }}
      />

      <div className="dashboard-container">
        <motion.div 
          className="header"
          initial={{ opacity: 0, y: -40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, type: 'spring', bounce: 0.4 }}
        >
          <h1><Sparkles size={48} /> VisaAI Pro</h1>
          <p>Next-Generation Processing Intelligence</p>
        </motion.div>

        <div className="metrics-grid">
          <MetricCard icon={<BarChart3 size={24} />} title="Analysis" value="Comprehensive EDA" delay={0.2} onClick={() => setActiveModal('analysis')} />
          <MetricCard icon={<Activity size={24} />} title="Logistics" value="Active Processing" delay={0.3} onClick={() => setActiveModal('logistics')} />
          <MetricCard icon={<Cpu size={24} />} title="Model Config" value="XGBoost Core" delay={0.4} onClick={() => setActiveModal('model')} />
        </div>

        <motion.div 
          className="glass-card"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
        >
          {loadingOptions ? (
            <motion.div className="loading-container" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <Loader2 className="spinner" size={64} />
              <h2>Initializing Core Models...</h2>
            </motion.div>
          ) : (
            <motion.form 
              onSubmit={handleSubmit}
              variants={containerVariants}
              initial="hidden"
              animate="show"
            >
              <div className="form-grid">
                {['case_status', 'visa_class', 'pw_wage_level', 'employer_city', 'worksite_city'].map((col) => (
                  <motion.div key={col} className="input-group" variants={itemVariants}>
                    <label>{col.replace('_', ' ')}</label>
                    <select className="input-control" name={col} value={formData[col]} onChange={handleChange} required>
                      {options?.[col]?.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                    </select>
                  </motion.div>
                ))}

                <motion.div className="input-group" variants={itemVariants}>
                  <label>APPLICATION TIMELINE</label>
                  <input className="input-control" type="date" name="app_date" value={formData.app_date} onChange={handleDateChange} required />
                </motion.div>

                <motion.div className="input-group" variants={itemVariants}>
                  <label>DECISION TIMELINE</label>
                  <input className="input-control" type="date" name="dec_date" value={formData.dec_date} onChange={handleDateChange} required />
                </motion.div>
              </div>

              <motion.button 
                type="submit" 
                className="submit-btn"
                disabled={isPredicting}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.96 }}
                variants={itemVariants}
              >
                {isPredicting ? <Loader2 className="spinner" size={28} /> : <><Rocket size={28} /> Predict Processing Time</>}
              </motion.button>
            </motion.form>
          )}
        </motion.div>

      <AppModal 
        isOpen={prediction !== null} 
        onClose={() => setPrediction(null)}
        title="Estimated Timeline"
      >
        <div style={{ textAlign: 'center' }}>
          <h2 style={{ fontSize: '5rem', margin: 0, background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-tertiary))', WebkitBackgroundClip: 'text', backgroundClip: 'text', color: 'transparent', fontWeight: 900, lineHeight: 1 }}>
            <AnimatedCounter value={prediction} />
          </h2>
          <div className="days-label">Total Days</div>
        </div>
      </AppModal>

      <AppModal isOpen={activeModal === 'analysis'} onClose={() => setActiveModal(null)} title="Data Analysis" large>
        <p>Historical H-1B Visa processing timeline trends spanning 2016 to 2023.</p>
        <div style={{ width: '100%', height: 350, marginTop: '2rem' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={analysisData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
              <XAxis dataKey="year" stroke="var(--text-secondary)" />
              <YAxis stroke="var(--text-secondary)" />
              <Tooltip 
                contentStyle={{ backgroundColor: 'var(--bg-secondary)', backdropFilter: 'blur(10px)', border: '1px solid var(--border-color)', borderRadius: '12px', color: 'var(--text-primary)' }}
                itemStyle={{ color: 'var(--accent-primary)', fontWeight: 'bold' }}
              />
              <Line type="monotone" dataKey="days" stroke="var(--accent-primary)" strokeWidth={3} dot={{ r: 5, fill: 'var(--accent-tertiary)' }} activeDot={{ r: 8 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </AppModal>

      <AppModal isOpen={activeModal === 'logistics'} onClose={() => setActiveModal(null)} title="Inference Logistics">
        <p>Real-time execution trace of the internal prediction pipeline layer.</p>
        <div style={{ marginTop: '2rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {[
            { id: 1, text: "Received JSON Request Payload", icon: <Database size={20} /> },
            { id: 2, text: "Sanitized & Normalized Input Parameters", icon: <Activity size={20} /> },
            { id: 3, text: "Allocated Resources in FastAPI Backend", icon: <Cpu size={20} /> },
            { id: 4, text: "Executed Extreme Gradient Boosting Algorithm", icon: <BarChart3 size={20} /> },
            { id: 5, text: prediction !== null ? `Successfully Forecasted: ${prediction} Processing Days` : 'Awaiting Model Output Initialization', icon: <Sparkles size={20} /> },
          ].map((step, index) => (
            <motion.div
               key={step.id}
               initial={{ opacity: 0, x: -30 }}
               animate={{ opacity: 1, x: 0 }}
               transition={{ delay: index * 0.2 + 0.1, type: "spring", stiffness: 100 }}
               style={{
                 display: 'flex', alignItems: 'center', gap: '1.25rem', 
                 background: 'var(--bg-secondary)', padding: '1.25rem', 
                 borderRadius: '12px', border: '1px solid var(--border-color)',
                 color: index === 4 && prediction !== null ? 'var(--accent-primary)' : 'var(--text-primary)',
                 fontWeight: index === 4 ? 'bold' : 'normal',
                 boxShadow: '0 4px 6px rgba(0,0,0,0.05)'
               }}
            >
              <div style={{ padding: '0.6rem', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '8px', color: 'var(--accent-primary)', flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {step.icon}
              </div>
              <span style={{ fontSize: '1.05rem' }}>{step.text}</span>
            </motion.div>
          ))}
        </div>
      </AppModal>

      <AppModal isOpen={activeModal === 'model'} onClose={() => setActiveModal(null)} title="Model Configuration" large>
        <p>Our algorithm executes highly optimized tree traversals to forecast processing trajectories.</p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem', marginTop: '2rem' }}>
          {[
            { id: 1, title: 'Architecture Layer', desc: 'XGBoost Regressor Algorithm targeting complex non-linear regression mappings.', icon: <Cpu size={28} /> },
            { id: 2, title: 'Hyperparameters', desc: 'Optimized Grid Search CV iterating through learning rates, max_depth, and n_estimators.', icon: <Activity size={28} /> },
            { id: 3, title: 'Loss Function', desc: 'Mean Absolute Error (MAE) utilized as the primary objective targeting mechanism.', icon: <BarChart3 size={28} /> }
          ].map((item, i) => (
             <motion.div
                key={item.id}
                initial={{ opacity: 0, scale: 0.6, y: 30 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                transition={{ delay: i * 0.2 + 0.1, type: "spring", stiffness: 100 }}
                style={{
                   background: 'var(--bg-secondary)', padding: '1.5rem', borderRadius: '16px', border: '1px solid var(--border-color)',
                   display: 'flex', flexDirection: 'column', gap: '0.75rem',
                   boxShadow: '0 8px 24px rgba(0,0,0,0.1)'
                }}
             >
                <div style={{ padding: '0.75rem', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '12px', color: 'var(--accent-primary)', width: 'fit-content' }}>
                   {item.icon}
                </div>
                <h4 style={{ margin: 0, color: 'var(--text-primary)', fontSize: '1.15rem' }}>{item.title}</h4>
                <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.4 }}>{item.desc}</p>
             </motion.div>
          ))}
        </div>

        <motion.div 
           initial={{ opacity: 0, y: 30 }}
           animate={{ opacity: 1, y: 0 }}
           transition={{ delay: 0.8, type: 'spring' }}
           style={{ marginTop: '2.5rem', background: 'var(--bg-secondary)', padding: '1.5rem', borderRadius: '16px', border: '1px solid var(--border-color)', boxShadow: 'inset 0 0 10px rgba(0,0,0,0.05)' }}
        >
          <h4 style={{ margin: '0 0 1rem 0', color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>XGBoost Hyperparameter Structure</h4>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem' }}>
            <span className="param-pill"><strong>n_estimators:</strong> 100</span>
            <span className="param-pill"><strong>n_jobs:</strong> -1</span>
            <span className="param-pill"><strong>objective:</strong> reg:squarederror</span>
            <span className="param-pill"><strong>booster:</strong> gbtree</span>
            <span className="param-pill"><strong>tree_method:</strong> auto</span>
          </div>
        </motion.div>
      </AppModal>
    </div>
  </>
  )
}

export default App
