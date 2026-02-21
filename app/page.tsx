'use client'

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { callAIAgent, extractText } from '@/lib/aiAgent'
import { getDocuments, uploadAndTrainDocument, deleteDocuments, crawlWebsite, validateFile } from '@/lib/ragKnowledgeBase'
import type { RAGDocument } from '@/lib/ragKnowledgeBase'
import { Button } from '@/components/ui/button'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Skeleton } from '@/components/ui/skeleton'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { HiOutlineChat, HiOutlineDocumentText, HiOutlineHome, HiOutlineClock, HiOutlineMenu, HiOutlineSearch, HiOutlineUpload, HiOutlineTrash, HiOutlineInformationCircle, HiOutlineLink, HiOutlineChevronRight, HiOutlineRefresh } from 'react-icons/hi'
import { FiSend, FiMessageSquare, FiDatabase, FiActivity, FiFileText, FiFile, FiGlobe, FiAlertTriangle, FiCheckCircle, FiXCircle } from 'react-icons/fi'

// ===== CONSTANTS =====
const AGENT_ID = '69997a7de6cce9ba73b56cca'
const RAG_ID = '69997a6ae12ce168202ff424'
const LOCAL_STORAGE_KEY = 'line-chatbot-conversations'

// ===== TYPES =====
interface ChatMessage {
  id: string
  role: 'user' | 'bot'
  content: string
  timestamp: string
  confidence?: 'high' | 'medium' | 'low'
  escalate?: boolean
  topic?: string
}

interface Conversation {
  id: string
  sessionId: string
  messages: ChatMessage[]
  startedAt: string
  lastMessageAt: string
}

type ActiveScreen = 'dashboard' | 'chat' | 'knowledge' | 'logs'

// ===== ERROR BOUNDARY =====
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: string }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props)
    this.state = { hasError: false, error: '' }
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error: error.message }
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-background text-foreground">
          <div className="text-center p-8 max-w-md">
            <h2 className="text-xl font-semibold mb-2">Something went wrong</h2>
            <p className="text-muted-foreground mb-4 text-sm">{this.state.error}</p>
            <button onClick={() => this.setState({ hasError: false, error: '' })} className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm">
              Try again
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

// ===== MARKDOWN RENDERER =====
function formatInline(text: string) {
  const parts = text.split(/\*\*(.*?)\*\*/g)
  if (parts.length === 1) return text
  return parts.map((part, i) =>
    i % 2 === 1 ? (
      <strong key={i} className="font-semibold">{part}</strong>
    ) : (
      part
    )
  )
}

function renderMarkdown(text: string) {
  if (!text) return null
  return (
    <div className="space-y-2">
      {text.split('\n').map((line, i) => {
        if (line.startsWith('### ')) return <h4 key={i} className="font-semibold text-sm mt-3 mb-1">{line.slice(4)}</h4>
        if (line.startsWith('## ')) return <h3 key={i} className="font-semibold text-base mt-3 mb-1">{line.slice(3)}</h3>
        if (line.startsWith('# ')) return <h2 key={i} className="font-bold text-lg mt-4 mb-2">{line.slice(2)}</h2>
        if (line.startsWith('- ') || line.startsWith('* ')) return <li key={i} className="ml-4 list-disc text-sm">{formatInline(line.slice(2))}</li>
        if (/^\d+\.\s/.test(line)) return <li key={i} className="ml-4 list-decimal text-sm">{formatInline(line.replace(/^\d+\.\s/, ''))}</li>
        if (!line.trim()) return <div key={i} className="h-1" />
        return <p key={i} className="text-sm">{formatInline(line)}</p>
      })}
    </div>
  )
}

// ===== HELPER: Generate unique ID =====
function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).substring(2, 9)
}

// ===== HELPER: Format time ago =====
function timeAgo(dateStr: string): string {
  const now = Date.now()
  const then = new Date(dateStr).getTime()
  const diffMs = now - then
  const diffSec = Math.floor(diffMs / 1000)
  if (diffSec < 60) return 'just now'
  const diffMin = Math.floor(diffSec / 60)
  if (diffMin < 60) return `${diffMin}m ago`
  const diffHr = Math.floor(diffMin / 60)
  if (diffHr < 24) return `${diffHr}h ago`
  const diffDay = Math.floor(diffHr / 24)
  return `${diffDay}d ago`
}

// ===== HELPER: Confidence color =====
function confidenceColor(conf?: string): string {
  switch (conf) {
    case 'high': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
    case 'medium': return 'bg-amber-500/20 text-amber-400 border-amber-500/30'
    case 'low': return 'bg-red-500/20 text-red-400 border-red-500/30'
    default: return 'bg-muted text-muted-foreground border-border'
  }
}

// ===== HELPER: localStorage for conversations =====
function loadConversations(): Conversation[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(LOCAL_STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function saveConversations(convos: Conversation[]) {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(convos))
  } catch {
    // localStorage may be full
  }
}

// ===== SAMPLE DATA =====
function getSampleConversations(): Conversation[] {
  const now = new Date()
  return [
    {
      id: 'sample-1',
      sessionId: 'sess-sample-1',
      startedAt: new Date(now.getTime() - 3600000).toISOString(),
      lastMessageAt: new Date(now.getTime() - 3000000).toISOString(),
      messages: [
        { id: 's1m1', role: 'user', content: 'What are your business hours?', timestamp: new Date(now.getTime() - 3600000).toISOString() },
        { id: 's1m2', role: 'bot', content: 'Our business hours are Monday to Friday, 9:00 AM to 6:00 PM JST. On weekends, we operate from 10:00 AM to 4:00 PM. Is there anything else I can help you with?', timestamp: new Date(now.getTime() - 3590000).toISOString(), confidence: 'high', topic: 'business_hours' },
        { id: 's1m3', role: 'user', content: 'Do you offer home delivery?', timestamp: new Date(now.getTime() - 3100000).toISOString() },
        { id: 's1m4', role: 'bot', content: 'Yes, we offer home delivery for orders above 2,000 JPY. Delivery is free within a 10km radius. For further distances, a small delivery fee applies. You can place your order through our LINE menu.', timestamp: new Date(now.getTime() - 3000000).toISOString(), confidence: 'high', topic: 'delivery' },
      ],
    },
    {
      id: 'sample-2',
      sessionId: 'sess-sample-2',
      startedAt: new Date(now.getTime() - 7200000).toISOString(),
      lastMessageAt: new Date(now.getTime() - 6800000).toISOString(),
      messages: [
        { id: 's2m1', role: 'user', content: 'I want to return a defective product', timestamp: new Date(now.getTime() - 7200000).toISOString() },
        { id: 's2m2', role: 'bot', content: 'I understand you received a defective product, and I sincerely apologize for the inconvenience. I recommend connecting you with our customer support team who can process your return quickly. They can be reached at support@example.com or through our returns portal.', timestamp: new Date(now.getTime() - 7100000).toISOString(), confidence: 'medium', escalate: true, topic: 'returns' },
        { id: 's2m3', role: 'user', content: 'Can you process the return for me?', timestamp: new Date(now.getTime() - 6900000).toISOString() },
        { id: 's2m4', role: 'bot', content: 'I am not able to directly process returns, but I can help connect you to a human agent who can handle this for you right away. Would you like me to do that?', timestamp: new Date(now.getTime() - 6800000).toISOString(), confidence: 'high', escalate: true, topic: 'returns' },
      ],
    },
    {
      id: 'sample-3',
      sessionId: 'sess-sample-3',
      startedAt: new Date(now.getTime() - 1800000).toISOString(),
      lastMessageAt: new Date(now.getTime() - 1500000).toISOString(),
      messages: [
        { id: 's3m1', role: 'user', content: 'What payment methods do you accept?', timestamp: new Date(now.getTime() - 1800000).toISOString() },
        { id: 's3m2', role: 'bot', content: 'We accept the following payment methods:\n- **Credit cards** (Visa, Mastercard, AMEX)\n- **LINE Pay**\n- **PayPay**\n- **Bank transfer**\n- **Cash on delivery** (for orders under 50,000 JPY)\n\nAll online payments are processed securely through our payment gateway.', timestamp: new Date(now.getTime() - 1700000).toISOString(), confidence: 'high', topic: 'payment' },
        { id: 's3m3', role: 'user', content: 'Do you support cryptocurrency?', timestamp: new Date(now.getTime() - 1600000).toISOString() },
        { id: 's3m4', role: 'bot', content: 'Currently, we do not accept cryptocurrency as a payment method. However, we are evaluating this option for the future. For now, I recommend using one of our existing payment options.', timestamp: new Date(now.getTime() - 1500000).toISOString(), confidence: 'low', topic: 'payment' },
      ],
    },
    {
      id: 'sample-4',
      sessionId: 'sess-sample-4',
      startedAt: new Date(now.getTime() - 10800000).toISOString(),
      lastMessageAt: new Date(now.getTime() - 10500000).toISOString(),
      messages: [
        { id: 's4m1', role: 'user', content: 'How can I track my order?', timestamp: new Date(now.getTime() - 10800000).toISOString() },
        { id: 's4m2', role: 'bot', content: 'You can track your order by visiting our tracking page and entering your order number. You should have received an order confirmation email with your tracking details. Alternatively, you can send me your order number and I can look it up for you.', timestamp: new Date(now.getTime() - 10500000).toISOString(), confidence: 'high', topic: 'order_tracking' },
      ],
    },
    {
      id: 'sample-5',
      sessionId: 'sess-sample-5',
      startedAt: new Date(now.getTime() - 14400000).toISOString(),
      lastMessageAt: new Date(now.getTime() - 14100000).toISOString(),
      messages: [
        { id: 's5m1', role: 'user', content: 'Do you have a loyalty program?', timestamp: new Date(now.getTime() - 14400000).toISOString() },
        { id: 's5m2', role: 'bot', content: 'Yes! Our LINE Loyalty Program rewards you for every purchase. You earn 1 point for every 100 JPY spent. Once you accumulate 500 points, you can redeem them for a 500 JPY discount on your next order. You can check your points balance anytime through our LINE menu.', timestamp: new Date(now.getTime() - 14100000).toISOString(), confidence: 'high', topic: 'loyalty_program' },
      ],
    },
  ]
}

// ===== SIDEBAR COMPONENT =====
function Sidebar({ activeScreen, setActiveScreen, sidebarOpen, setSidebarOpen }: {
  activeScreen: ActiveScreen
  setActiveScreen: (s: ActiveScreen) => void
  sidebarOpen: boolean
  setSidebarOpen: (b: boolean) => void
}) {
  const navItems: { id: ActiveScreen; label: string; icon: React.ReactNode }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: <HiOutlineHome className="w-5 h-5" /> },
    { id: 'chat', label: 'Chat Preview', icon: <HiOutlineChat className="w-5 h-5" /> },
    { id: 'knowledge', label: 'Knowledge Base', icon: <HiOutlineDocumentText className="w-5 h-5" /> },
    { id: 'logs', label: 'Conversation Logs', icon: <HiOutlineClock className="w-5 h-5" /> },
  ]

  return (
    <>
      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/50 z-30 lg:hidden" onClick={() => setSidebarOpen(false)} />
      )}
      <aside className={`fixed top-0 left-0 z-40 h-full w-[260px] bg-card border-r border-border flex flex-col transition-transform duration-300 lg:translate-x-0 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="flex items-center gap-3 px-5 py-5 border-b border-border">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: 'hsl(160, 70%, 40%)' }}>
            <FiMessageSquare className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-base font-bold text-foreground tracking-tight">LINE Chatbot</h1>
            <p className="text-[11px] text-muted-foreground">Admin Panel</p>
          </div>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => { setActiveScreen(item.id); setSidebarOpen(false) }}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${activeScreen === item.id ? 'text-accent-foreground' : 'text-muted-foreground hover:text-foreground hover:bg-secondary'}`}
              style={activeScreen === item.id ? { background: 'hsl(160, 70%, 40%)', color: 'hsl(160, 20%, 98%)' } : {}}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </nav>

        <div className="px-4 py-4 border-t border-border">
          <div className="flex items-center gap-2 px-2">
            <div className="w-2 h-2 rounded-full" style={{ background: 'hsl(160, 70%, 40%)' }} />
            <span className="text-xs text-muted-foreground">Agent Online</span>
          </div>
          <p className="text-[10px] text-muted-foreground mt-1 px-2">ID: {AGENT_ID.slice(0, 12)}...</p>
        </div>
      </aside>
    </>
  )
}

// ===== HEADER COMPONENT =====
function Header({ title, setSidebarOpen }: { title: string; setSidebarOpen: (b: boolean) => void }) {
  return (
    <header className="sticky top-0 z-20 bg-card/80 backdrop-blur-xl border-b border-border px-4 lg:px-6 py-3 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <button onClick={() => setSidebarOpen(true)} className="lg:hidden p-1.5 rounded-lg hover:bg-secondary transition-colors">
          <HiOutlineMenu className="w-5 h-5 text-foreground" />
        </button>
        <h2 className="text-lg font-semibold text-foreground">{title}</h2>
      </div>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Badge variant="outline" className="bg-amber-500/10 text-amber-400 border-amber-500/30 cursor-help gap-1.5 px-3 py-1">
              <HiOutlineInformationCircle className="w-3.5 h-3.5" />
              Setup Required
            </Badge>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="max-w-xs bg-popover text-popover-foreground">
            <p className="text-sm">To connect to LINE, configure your Messaging API webhook URL in the LINE Developers Console to point to your production server endpoint.</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </header>
  )
}

// ===== STAT CARD =====
function StatCard({ label, value, icon, accent }: { label: string; value: string | number; icon: React.ReactNode; accent?: boolean }) {
  return (
    <Card className="bg-card border-border">
      <CardContent className="p-5">
        <div className="flex items-center justify-between mb-3">
          <span className="text-muted-foreground text-xs font-medium uppercase tracking-wider">{label}</span>
          <div className={`w-9 h-9 rounded-lg flex items-center justify-center ${accent ? '' : 'bg-secondary'}`} style={accent ? { background: 'hsl(160, 70%, 40%)' } : {}}>
            {icon}
          </div>
        </div>
        <p className="text-2xl font-bold text-foreground">{value}</p>
      </CardContent>
    </Card>
  )
}

// ===== DASHBOARD SCREEN =====
function DashboardScreen({ conversations, docCount, docLoading, setActiveScreen, sampleMode }: {
  conversations: Conversation[]
  docCount: number
  docLoading: boolean
  setActiveScreen: (s: ActiveScreen) => void
  sampleMode: boolean
}) {
  const displayConvos = sampleMode && conversations.length === 0 ? getSampleConversations() : conversations
  const recentConvos = displayConvos.slice(-5).reverse()
  const totalMessages = displayConvos.reduce((acc, c) => acc + (Array.isArray(c?.messages) ? c.messages.length : 0), 0)

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Conversations" value={displayConvos.length} icon={<FiMessageSquare className="w-4 h-4 text-foreground" />} />
        <StatCard label="Total Messages" value={totalMessages} icon={<FiActivity className="w-4 h-4 text-white" />} accent />
        <StatCard label="KB Documents" value={docLoading ? '...' : docCount} icon={<FiDatabase className="w-4 h-4 text-foreground" />} />
        <StatCard label="LINE Status" value="Pending" icon={<FiAlertTriangle className="w-4 h-4 text-amber-400" />} />
      </div>

      <Card className="bg-card border-border">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base font-semibold">Recent Conversations</CardTitle>
            <Button variant="ghost" size="sm" className="text-xs text-muted-foreground hover:text-foreground gap-1" onClick={() => setActiveScreen('logs')}>
              View All <HiOutlineChevronRight className="w-3.5 h-3.5" />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {recentConvos.length === 0 ? (
            <div className="py-12 text-center">
              <FiMessageSquare className="w-10 h-10 text-muted-foreground mx-auto mb-3 opacity-40" />
              <p className="text-sm text-muted-foreground">No conversations yet</p>
              <p className="text-xs text-muted-foreground mt-1">Start a chat in Chat Preview to see activity here</p>
            </div>
          ) : (
            <div>
              {recentConvos.map((convo, idx) => {
                const msgs = Array.isArray(convo?.messages) ? convo.messages : []
                const firstUser = msgs.find(m => m?.role === 'user')
                const lastBot = [...msgs].reverse().find(m => m?.role === 'bot')
                return (
                  <div key={convo?.id ?? idx} className={`px-5 py-3.5 flex items-start gap-4 hover:bg-secondary/50 transition-colors cursor-pointer ${idx < recentConvos.length - 1 ? 'border-b border-border' : ''}`} onClick={() => setActiveScreen('logs')}>
                    <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center flex-shrink-0 mt-0.5">
                      <HiOutlineChat className="w-4 h-4 text-muted-foreground" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-foreground truncate">{firstUser?.content ?? 'No message'}</p>
                      <p className="text-xs text-muted-foreground truncate mt-0.5">{lastBot?.content ?? '...'}</p>
                    </div>
                    <div className="flex flex-col items-end gap-1 flex-shrink-0">
                      <span className="text-[10px] text-muted-foreground">{convo?.lastMessageAt ? timeAgo(convo.lastMessageAt) : ''}</span>
                      {lastBot?.confidence && (
                        <Badge variant="outline" className={`text-[10px] px-1.5 py-0 ${confidenceColor(lastBot.confidence)}`}>
                          {lastBot.confidence}
                        </Badge>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </CardContent>
      </Card>

      <div className="flex justify-center">
        <Button onClick={() => setActiveScreen('chat')} className="gap-2 px-6" style={{ background: 'hsl(160, 70%, 40%)', color: 'white' }}>
          <HiOutlineChat className="w-4 h-4" />
          Go to Chat Preview
        </Button>
      </div>
    </div>
  )
}

// ===== TYPING INDICATOR =====
function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 px-4 py-3">
      <div className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce" style={{ animationDelay: '0ms' }} />
      <div className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce" style={{ animationDelay: '150ms' }} />
      <div className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce" style={{ animationDelay: '300ms' }} />
    </div>
  )
}

// ===== CHAT SCREEN =====
function ChatScreen({ conversations, setConversations, sampleMode, setActiveAgentId }: {
  conversations: Conversation[]
  setConversations: React.Dispatch<React.SetStateAction<Conversation[]>>
  sampleMode: boolean
  setActiveAgentId: (id: string | null) => void
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sessionId] = useState(() => generateId())
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (sampleMode && messages.length === 0) {
      const sample = getSampleConversations()[0]
      if (sample && Array.isArray(sample.messages)) {
        setMessages(sample.messages)
      }
    }
  }, [sampleMode, messages.length])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, loading])

  const saveCurrentConversation = useCallback((msgs: ChatMessage[]) => {
    if (msgs.length === 0) return
    const now = new Date().toISOString()
    setConversations(prev => {
      const existingIdx = prev.findIndex(c => c.sessionId === sessionId)
      const convo: Conversation = {
        id: existingIdx >= 0 ? prev[existingIdx].id : generateId(),
        sessionId,
        messages: msgs,
        startedAt: existingIdx >= 0 ? prev[existingIdx].startedAt : msgs[0]?.timestamp ?? now,
        lastMessageAt: msgs[msgs.length - 1]?.timestamp ?? now,
      }
      const updated = existingIdx >= 0
        ? prev.map((c, i) => i === existingIdx ? convo : c)
        : [...prev, convo]
      saveConversations(updated)
      return updated
    })
  }, [sessionId, setConversations])

  const handleSend = async () => {
    const trimmed = input.trim()
    if (!trimmed || loading) return

    setError(null)
    const userMsg: ChatMessage = {
      id: generateId(),
      role: 'user',
      content: trimmed,
      timestamp: new Date().toISOString(),
    }

    const updatedMsgs = [...messages, userMsg]
    setMessages(updatedMsgs)
    setInput('')
    setLoading(true)
    setActiveAgentId(AGENT_ID)

    try {
      const result = await callAIAgent(trimmed, AGENT_ID, { session_id: sessionId })
      if (result.success && result?.response?.result) {
        let parsedResult = result.response.result
        if (typeof parsedResult === 'string') {
          try { parsedResult = JSON.parse(parsedResult) } catch { parsedResult = { response: parsedResult } }
        }
        const botContent = parsedResult?.response || extractText(result.response) || 'Sorry, I could not process that.'
        const confidence = parsedResult?.confidence || 'medium'
        const escalate = parsedResult?.escalate || false
        const topic = parsedResult?.topic || 'general'

        const botMsg: ChatMessage = {
          id: generateId(),
          role: 'bot',
          content: typeof botContent === 'string' ? botContent : String(botContent),
          timestamp: new Date().toISOString(),
          confidence,
          escalate,
          topic,
        }
        const finalMsgs = [...updatedMsgs, botMsg]
        setMessages(finalMsgs)
        saveCurrentConversation(finalMsgs)
      } else {
        const errMsg = result?.error || result?.response?.message || 'Failed to get response'
        setError(typeof errMsg === 'string' ? errMsg : 'Failed to get response')
        saveCurrentConversation(updatedMsgs)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error')
      saveCurrentConversation(updatedMsgs)
    } finally {
      setLoading(false)
      setActiveAgentId(null)
      inputRef.current?.focus()
    }
  }

  const handleClear = () => {
    setMessages([])
    setError(null)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <Card className="bg-card border-border flex flex-col h-[calc(100vh-8rem)]">
      <CardHeader className="py-3 px-4 border-b border-border flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-full flex items-center justify-center" style={{ background: 'hsl(160, 70%, 40%)' }}>
              <FiMessageSquare className="w-4 h-4 text-white" />
            </div>
            <div>
              <CardTitle className="text-sm font-semibold">Chat Preview</CardTitle>
              <p className="text-[10px] text-muted-foreground">Session: {sessionId.slice(0, 8)}</p>
            </div>
          </div>
          <Button variant="ghost" size="sm" className="text-xs text-muted-foreground hover:text-foreground" onClick={handleClear}>
            <HiOutlineRefresh className="w-3.5 h-3.5 mr-1" /> Clear
          </Button>
        </div>
      </CardHeader>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center h-full text-center py-16">
            <div className="w-14 h-14 rounded-2xl flex items-center justify-center mb-4 bg-secondary">
              <HiOutlineChat className="w-7 h-7 text-muted-foreground" />
            </div>
            <p className="text-sm text-foreground font-medium">Start a conversation</p>
            <p className="text-xs text-muted-foreground mt-1 max-w-[280px]">Type a message below to test the LINE Customer Chat Agent. The agent uses your knowledge base to answer questions.</p>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] ${msg.role === 'user' ? '' : ''}`}>
              {msg.role === 'bot' && msg.escalate && (
                <div className="flex items-center gap-1.5 mb-1.5 px-1">
                  <FiAlertTriangle className="w-3 h-3 text-amber-400" />
                  <span className="text-[10px] text-amber-400 font-medium">This query may need human assistance</span>
                </div>
              )}
              <div className={`rounded-2xl px-4 py-2.5 text-sm ${msg.role === 'user' ? 'rounded-br-md text-white' : 'bg-secondary rounded-bl-md text-foreground'}`} style={msg.role === 'user' ? { background: 'hsl(160, 70%, 40%)' } : {}}>
                {msg.role === 'bot' ? renderMarkdown(msg.content) : msg.content}
              </div>
              <div className={`flex items-center gap-2 mt-1 px-1 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <span className="text-[10px] text-muted-foreground">{msg.timestamp ? timeAgo(msg.timestamp) : ''}</span>
                {msg.role === 'bot' && msg.confidence && (
                  <Badge variant="outline" className={`text-[9px] px-1.5 py-0 h-4 ${confidenceColor(msg.confidence)}`}>
                    {msg.confidence}
                  </Badge>
                )}
                {msg.role === 'bot' && msg.topic && (
                  <Badge variant="outline" className="text-[9px] px-1.5 py-0 h-4 bg-secondary text-muted-foreground border-border">
                    {msg.topic}
                  </Badge>
                )}
              </div>
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-secondary rounded-2xl rounded-bl-md">
              <TypingIndicator />
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="px-4 py-2 border-t border-border">
          <p className="text-xs text-red-400 flex items-center gap-1.5">
            <FiXCircle className="w-3 h-3" /> {error}
          </p>
        </div>
      )}

      <div className="px-4 py-3 border-t border-border flex-shrink-0">
        <div className="flex items-center gap-2">
          <Input
            ref={inputRef}
            placeholder="Type a message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            className="flex-1 bg-secondary border-border text-foreground placeholder:text-muted-foreground"
          />
          <Button onClick={handleSend} disabled={loading || !input.trim()} size="sm" className="px-3 h-9" style={{ background: 'hsl(160, 70%, 40%)', color: 'white' }}>
            <FiSend className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </Card>
  )
}

// ===== KNOWLEDGE BASE SCREEN =====
function KnowledgeBaseScreen({ docCount, setDocCount }: { docCount: number; setDocCount: (n: number) => void }) {
  const [documents, setDocuments] = useState<RAGDocument[]>([])
  const [loading, setLoading] = useState(true)
  const [uploadStatus, setUploadStatus] = useState<{ type: 'success' | 'error' | 'loading'; message: string } | null>(null)
  const [crawlUrl, setCrawlUrl] = useState('')
  const [crawlStatus, setCrawlStatus] = useState<{ type: 'success' | 'error' | 'loading'; message: string } | null>(null)
  const [deleteStatus, setDeleteStatus] = useState<{ [key: string]: string }>({})
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const fetchDocs = useCallback(async () => {
    setLoading(true)
    const res = await getDocuments(RAG_ID)
    if (res.success && Array.isArray(res.documents)) {
      setDocuments(res.documents)
      setDocCount(res.documents.length)
    } else {
      setDocuments([])
      setDocCount(0)
    }
    setLoading(false)
  }, [setDocCount])

  useEffect(() => {
    fetchDocs()
  }, [fetchDocs])

  const handleFileUpload = async (file: File) => {
    const validation = validateFile(file)
    if (!validation.valid) {
      setUploadStatus({ type: 'error', message: validation.error ?? 'Invalid file type' })
      return
    }
    setUploadStatus({ type: 'loading', message: `Uploading ${file.name}...` })
    const res = await uploadAndTrainDocument(RAG_ID, file)
    if (res.success) {
      setUploadStatus({ type: 'success', message: `${file.name} uploaded and trained successfully` })
      await fetchDocs()
    } else {
      setUploadStatus({ type: 'error', message: res.error ?? 'Upload failed' })
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleFileUpload(file)
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFileUpload(file)
    if (e.target) e.target.value = ''
  }

  const handleCrawl = async () => {
    const trimmedUrl = crawlUrl.trim()
    if (!trimmedUrl) return
    setCrawlStatus({ type: 'loading', message: `Crawling ${trimmedUrl}...` })
    const res = await crawlWebsite(RAG_ID, trimmedUrl)
    if (res.success) {
      setCrawlStatus({ type: 'success', message: res.message ?? 'Website crawled successfully' })
      setCrawlUrl('')
      await fetchDocs()
    } else {
      setCrawlStatus({ type: 'error', message: res.error ?? 'Crawl failed' })
    }
  }

  const handleDelete = async (fileName: string) => {
    setDeleteStatus(prev => ({ ...prev, [fileName]: 'deleting' }))
    const res = await deleteDocuments(RAG_ID, [fileName])
    if (res.success) {
      setDocuments(prev => prev.filter(d => d.fileName !== fileName))
      setDocCount(Math.max(0, docCount - 1))
      setDeleteStatus(prev => {
        const next = { ...prev }
        delete next[fileName]
        return next
      })
    } else {
      setDeleteStatus(prev => ({ ...prev, [fileName]: res.error ?? 'Delete failed' }))
    }
  }

  const fileTypeIcon = (ft?: string) => {
    switch (ft) {
      case 'pdf': return <FiFileText className="w-5 h-5 text-red-400" />
      case 'docx': return <FiFile className="w-5 h-5 text-blue-400" />
      case 'csv': return <FiFileText className="w-5 h-5 text-emerald-400" />
      case 'txt': return <FiFileText className="w-5 h-5 text-muted-foreground" />
      default: return <FiFile className="w-5 h-5 text-muted-foreground" />
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Upload Section */}
        <Card className="bg-card border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <HiOutlineUpload className="w-4 h-4" /> Upload Document
            </CardTitle>
            <CardDescription className="text-xs">PDF, DOCX, TXT, or CSV files</CardDescription>
          </CardHeader>
          <CardContent>
            <div
              className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 ${dragOver ? 'border-ring bg-secondary/80' : 'border-border hover:border-muted-foreground hover:bg-secondary/30'}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <HiOutlineUpload className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
              <p className="text-sm text-foreground font-medium">Drop file here or click to browse</p>
              <p className="text-xs text-muted-foreground mt-1">Supports PDF, DOCX, TXT, CSV</p>
              <input ref={fileInputRef} type="file" className="hidden" accept=".pdf,.docx,.txt,.csv" onChange={handleFileSelect} />
            </div>
            {uploadStatus && (
              <div className={`mt-3 flex items-center gap-2 text-xs px-3 py-2 rounded-lg ${uploadStatus.type === 'success' ? 'bg-emerald-500/10 text-emerald-400' : uploadStatus.type === 'error' ? 'bg-red-500/10 text-red-400' : 'bg-secondary text-muted-foreground'}`}>
                {uploadStatus.type === 'loading' && <div className="w-3 h-3 border-2 border-muted-foreground border-t-transparent rounded-full animate-spin" />}
                {uploadStatus.type === 'success' && <FiCheckCircle className="w-3 h-3" />}
                {uploadStatus.type === 'error' && <FiXCircle className="w-3 h-3" />}
                {uploadStatus.message}
              </div>
            )}
          </CardContent>
        </Card>

        {/* URL Crawl Section */}
        <Card className="bg-card border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <FiGlobe className="w-4 h-4" /> Crawl Website
            </CardTitle>
            <CardDescription className="text-xs">Add web content to knowledge base</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex gap-2">
              <Input
                placeholder="https://example.com"
                value={crawlUrl}
                onChange={(e) => setCrawlUrl(e.target.value)}
                className="flex-1 bg-secondary border-border text-foreground placeholder:text-muted-foreground"
              />
              <Button onClick={handleCrawl} disabled={!crawlUrl.trim() || crawlStatus?.type === 'loading'} size="sm" className="px-4" style={{ background: 'hsl(160, 70%, 40%)', color: 'white' }}>
                {crawlStatus?.type === 'loading' ? (
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  <HiOutlineLink className="w-4 h-4" />
                )}
              </Button>
            </div>
            {crawlStatus && (
              <div className={`mt-3 flex items-center gap-2 text-xs px-3 py-2 rounded-lg ${crawlStatus.type === 'success' ? 'bg-emerald-500/10 text-emerald-400' : crawlStatus.type === 'error' ? 'bg-red-500/10 text-red-400' : 'bg-secondary text-muted-foreground'}`}>
                {crawlStatus.type === 'loading' && <div className="w-3 h-3 border-2 border-muted-foreground border-t-transparent rounded-full animate-spin" />}
                {crawlStatus.type === 'success' && <FiCheckCircle className="w-3 h-3" />}
                {crawlStatus.type === 'error' && <FiXCircle className="w-3 h-3" />}
                {crawlStatus.message}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Documents Grid */}
      <Card className="bg-card border-border">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-semibold">Documents ({documents.length})</CardTitle>
            <Button variant="ghost" size="sm" className="text-xs text-muted-foreground" onClick={fetchDocs}>
              <HiOutlineRefresh className="w-3.5 h-3.5 mr-1" /> Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {[1, 2, 3].map(i => (
                <div key={i} className="p-4 rounded-lg bg-secondary animate-pulse">
                  <Skeleton className="h-5 w-5 rounded mb-3" />
                  <Skeleton className="h-4 w-3/4 rounded mb-2" />
                  <Skeleton className="h-3 w-1/2 rounded" />
                </div>
              ))}
            </div>
          ) : documents.length === 0 ? (
            <div className="py-12 text-center">
              <FiDatabase className="w-12 h-12 text-muted-foreground mx-auto mb-3 opacity-30" />
              <p className="text-sm text-foreground font-medium">No documents yet</p>
              <p className="text-xs text-muted-foreground mt-1">Upload your first document to get started</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {documents.map((doc) => (
                <div key={doc?.fileName ?? generateId()} className="p-4 rounded-lg bg-secondary/50 border border-border hover:bg-secondary transition-colors group">
                  <div className="flex items-start justify-between">
                    {fileTypeIcon(doc?.fileType)}
                    <Button
                      variant="ghost"
                      size="sm"
                      className="opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-red-400 h-7 w-7 p-0"
                      onClick={() => doc?.fileName && handleDelete(doc.fileName)}
                      disabled={deleteStatus[doc?.fileName ?? ''] === 'deleting'}
                    >
                      {deleteStatus[doc?.fileName ?? ''] === 'deleting' ? (
                        <div className="w-3 h-3 border-2 border-muted-foreground border-t-transparent rounded-full animate-spin" />
                      ) : (
                        <HiOutlineTrash className="w-3.5 h-3.5" />
                      )}
                    </Button>
                  </div>
                  <p className="text-sm text-foreground font-medium mt-2 truncate">{doc?.fileName ?? 'Unknown'}</p>
                  <div className="flex items-center gap-2 mt-1">
                    <Badge variant="outline" className="text-[10px] px-1.5 py-0 bg-secondary text-muted-foreground border-border uppercase">{doc?.fileType ?? 'file'}</Badge>
                    {doc?.uploadedAt && <span className="text-[10px] text-muted-foreground">{timeAgo(doc.uploadedAt)}</span>}
                  </div>
                  {typeof deleteStatus[doc?.fileName ?? ''] === 'string' && deleteStatus[doc?.fileName ?? ''] !== 'deleting' && (
                    <p className="text-[10px] text-red-400 mt-1">{deleteStatus[doc?.fileName ?? '']}</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

// ===== CONVERSATION LOGS SCREEN =====
function ConversationLogsScreen({ conversations, sampleMode }: { conversations: Conversation[]; sampleMode: boolean }) {
  const displayConvos = sampleMode && conversations.length === 0 ? getSampleConversations() : conversations
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')

  const filteredConvos = useMemo(() => {
    if (!searchQuery.trim()) return displayConvos
    const q = searchQuery.toLowerCase()
    return displayConvos.filter(c => {
      const msgs = Array.isArray(c?.messages) ? c.messages : []
      return msgs.some(m => (m?.content ?? '').toLowerCase().includes(q))
    })
  }, [displayConvos, searchQuery])

  const selectedConvo = useMemo(() => {
    if (!selectedId) return filteredConvos[0] ?? null
    return filteredConvos.find(c => c.id === selectedId) ?? filteredConvos[0] ?? null
  }, [filteredConvos, selectedId])

  const selectedMessages = Array.isArray(selectedConvo?.messages) ? selectedConvo.messages : []

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 h-[calc(100vh-8rem)]">
      {/* Left: Conversation List */}
      <Card className="bg-card border-border lg:col-span-1 flex flex-col overflow-hidden">
        <CardHeader className="py-3 px-4 border-b border-border flex-shrink-0">
          <div className="relative">
            <HiOutlineSearch className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search conversations..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 bg-secondary border-border text-foreground placeholder:text-muted-foreground h-8 text-xs"
            />
          </div>
        </CardHeader>
        <ScrollArea className="flex-1">
          {filteredConvos.length === 0 ? (
            <div className="py-12 text-center px-4">
              <HiOutlineClock className="w-10 h-10 text-muted-foreground mx-auto mb-3 opacity-30" />
              <p className="text-sm text-foreground font-medium">No conversations yet</p>
              <p className="text-xs text-muted-foreground mt-1">Try the Chat Preview to get started</p>
            </div>
          ) : (
            <div>
              {filteredConvos.map((convo, idx) => {
                const msgs = Array.isArray(convo?.messages) ? convo.messages : []
                const firstUser = msgs.find(m => m?.role === 'user')
                const topics = [...new Set(msgs.filter(m => m?.topic).map(m => m.topic))]
                const isSelected = (selectedConvo?.id ?? '') === (convo?.id ?? '')
                return (
                  <div
                    key={convo?.id ?? idx}
                    className={`px-4 py-3 cursor-pointer transition-colors border-b border-border ${isSelected ? 'bg-secondary' : 'hover:bg-secondary/50'}`}
                    onClick={() => setSelectedId(convo?.id ?? null)}
                  >
                    <p className="text-sm text-foreground truncate font-medium">{firstUser?.content ?? 'Empty conversation'}</p>
                    <div className="flex items-center gap-2 mt-1.5">
                      <span className="text-[10px] text-muted-foreground">{msgs.length} msgs</span>
                      <span className="text-[10px] text-muted-foreground">{convo?.lastMessageAt ? timeAgo(convo.lastMessageAt) : ''}</span>
                    </div>
                    {topics.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1.5">
                        {topics.slice(0, 3).map((t, ti) => (
                          <Badge key={ti} variant="outline" className="text-[9px] px-1.5 py-0 h-4 bg-secondary text-muted-foreground border-border">
                            {t}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </ScrollArea>
      </Card>

      {/* Right: Conversation Detail */}
      <Card className="bg-card border-border lg:col-span-2 flex flex-col overflow-hidden">
        {selectedConvo ? (
          <>
            <CardHeader className="py-3 px-4 border-b border-border flex-shrink-0">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-sm font-semibold">Conversation Detail</CardTitle>
                  <p className="text-[10px] text-muted-foreground mt-0.5">Session: {selectedConvo.sessionId?.slice(0, 12) ?? '...'} | {selectedMessages.length} messages</p>
                </div>
                <Badge variant="outline" className="text-[10px] px-2 py-0.5 bg-secondary text-muted-foreground border-border">
                  {selectedConvo.startedAt ? new Date(selectedConvo.startedAt).toLocaleDateString() : ''}
                </Badge>
              </div>
            </CardHeader>
            <ScrollArea className="flex-1 px-4 py-4">
              <div className="space-y-4">
                {selectedMessages.map((msg) => (
                  <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className="max-w-[80%]">
                      {msg.role === 'bot' && msg.escalate && (
                        <div className="flex items-center gap-1.5 mb-1.5 px-1">
                          <FiAlertTriangle className="w-3 h-3 text-amber-400" />
                          <span className="text-[10px] text-amber-400 font-medium">Escalation suggested</span>
                        </div>
                      )}
                      <div className={`rounded-2xl px-4 py-2.5 text-sm ${msg.role === 'user' ? 'rounded-br-md text-white' : 'bg-secondary rounded-bl-md text-foreground'}`} style={msg.role === 'user' ? { background: 'hsl(160, 70%, 40%)' } : {}}>
                        {msg.role === 'bot' ? renderMarkdown(msg.content) : msg.content}
                      </div>
                      <div className={`flex items-center gap-2 mt-1 px-1 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <span className="text-[10px] text-muted-foreground">{msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString() : ''}</span>
                        {msg.role === 'bot' && msg.confidence && (
                          <Badge variant="outline" className={`text-[9px] px-1.5 py-0 h-4 ${confidenceColor(msg.confidence)}`}>
                            {msg.confidence}
                          </Badge>
                        )}
                        {msg.role === 'bot' && msg.topic && (
                          <Badge variant="outline" className="text-[9px] px-1.5 py-0 h-4 bg-secondary text-muted-foreground border-border">
                            {msg.topic}
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <HiOutlineChat className="w-12 h-12 text-muted-foreground mx-auto mb-3 opacity-30" />
              <p className="text-sm text-foreground font-medium">No conversation selected</p>
              <p className="text-xs text-muted-foreground mt-1">Select a conversation from the list to view details</p>
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}

// ===== AGENT STATUS COMPONENT =====
function AgentStatusPanel({ activeAgentId }: { activeAgentId: string | null }) {
  return (
    <Card className="bg-card border-border mt-6">
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: 'hsl(160, 70%, 40%)' }}>
              <FiActivity className="w-3.5 h-3.5 text-white" />
            </div>
            <div>
              <p className="text-xs font-semibold text-foreground">LINE Customer Chat Agent</p>
              <p className="text-[10px] text-muted-foreground">Handles customer conversations, answers from KB, escalates when needed</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {activeAgentId === AGENT_ID ? (
              <Badge className="text-[10px] px-2 py-0.5 gap-1 bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Active
              </Badge>
            ) : (
              <Badge variant="outline" className="text-[10px] px-2 py-0.5 gap-1 text-muted-foreground border-border">
                <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground" />
                Idle
              </Badge>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// ===== MAIN PAGE COMPONENT =====
export default function Page() {
  const [activeScreen, setActiveScreen] = useState<ActiveScreen>('dashboard')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sampleMode, setSampleMode] = useState(false)
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [docCount, setDocCount] = useState(0)
  const [docLoading, setDocLoading] = useState(true)
  const [activeAgentId, setActiveAgentId] = useState<string | null>(null)

  // Load conversations from localStorage on mount
  useEffect(() => {
    setConversations(loadConversations())
  }, [])

  // Fetch doc count on mount
  useEffect(() => {
    async function fetchDocCount() {
      setDocLoading(true)
      const res = await getDocuments(RAG_ID)
      if (res.success && Array.isArray(res.documents)) {
        setDocCount(res.documents.length)
      }
      setDocLoading(false)
    }
    fetchDocCount()
  }, [])

  const screenTitles: Record<ActiveScreen, string> = {
    dashboard: 'Dashboard',
    chat: 'Chat Preview',
    knowledge: 'Knowledge Base',
    logs: 'Conversation Logs',
  }

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-background text-foreground">
        <Sidebar activeScreen={activeScreen} setActiveScreen={setActiveScreen} sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />

        <div className="lg:ml-[260px] min-h-screen flex flex-col">
          <Header title={screenTitles[activeScreen]} setSidebarOpen={setSidebarOpen} />

          <main className="flex-1 p-4 lg:p-6">
            {/* Sample Data Toggle */}
            <div className="flex items-center justify-end mb-4">
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <span className="text-xs text-muted-foreground">Sample Data</span>
                <button
                  onClick={() => setSampleMode(!sampleMode)}
                  className={`relative w-9 h-5 rounded-full transition-colors duration-200 ${sampleMode ? '' : 'bg-secondary'}`}
                  style={sampleMode ? { background: 'hsl(160, 70%, 40%)' } : {}}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform duration-200 ${sampleMode ? 'translate-x-4' : 'translate-x-0'}`} />
                </button>
              </label>
            </div>

            {activeScreen === 'dashboard' && (
              <DashboardScreen
                conversations={conversations}
                docCount={docCount}
                docLoading={docLoading}
                setActiveScreen={setActiveScreen}
                sampleMode={sampleMode}
              />
            )}

            {activeScreen === 'chat' && (
              <ChatScreen
                conversations={conversations}
                setConversations={setConversations}
                sampleMode={sampleMode}
                setActiveAgentId={setActiveAgentId}
              />
            )}

            {activeScreen === 'knowledge' && (
              <KnowledgeBaseScreen docCount={docCount} setDocCount={setDocCount} />
            )}

            {activeScreen === 'logs' && (
              <ConversationLogsScreen conversations={conversations} sampleMode={sampleMode} />
            )}

            <AgentStatusPanel activeAgentId={activeAgentId} />
          </main>
        </div>
      </div>
    </ErrorBoundary>
  )
}
