// ── FAQ Database ──────────────────────────────
var faqs = [
  {
    question: "Who are the host countries?",
    keywords: ["host", "country", "countries", "hosting", "where"],
    answer: "🌎 FIFA World Cup 2026 is hosted by three countries: USA, Canada, and Mexico. This is the first time three nations have co-hosted the World Cup!"
  },
  {
    question: "How many teams are participating?",
    keywords: ["teams", "how many", "participating", "countries", "nations", "qualify"],
    answer: "👥 48 teams are participating in FIFA World Cup 2026 — up from 32 in previous tournaments. This is the largest World Cup in history!"
  },
  {
    question: "When does the tournament start?",
    keywords: ["when", "start", "begin", "date", "schedule", "opening"],
    answer: "📅 FIFA World Cup 2026 starts on June 11, 2026, with the final on July 19, 2026. The opening match is in Mexico City!"
  },
  {
    question: "Which stadiums are being used?",
    keywords: ["stadium", "stadiums", "venue", "venues", "arena", "where played"],
    answer: "🏟️ 16 stadiums across 3 countries will host matches:\n🇺🇸 USA: New York, Los Angeles, Dallas, San Francisco, Miami, Seattle, Boston, Kansas City, Atlanta, Philadelphia\n🇨🇦 Canada: Toronto, Vancouver\n🇲🇽 Mexico: Mexico City, Guadalajara, Monterrey"
  },
  {
    question: "How does the group stage work?",
    keywords: ["group", "stage", "format", "how", "works", "round", "phase"],
    answer: "📋 In 2026, there are 12 groups of 4 teams each. The top 2 teams from each group plus 8 best third-place teams (24 total) advance to the Round of 32!"
  },
  {
    question: "What is VAR?",
    keywords: ["var", "video", "referee", "review", "technology", "decision"],
    answer: "📺 VAR (Video Assistant Referee) is a technology used to review key match decisions like goals, penalties, red cards, and mistaken identity using video footage."
  },
  {
    question: "Who won the last World Cup?",
    keywords: ["last", "previous", "won", "winner", "2022", "champion", "argentina"],
    answer: "🏆 Argentina won the 2022 FIFA World Cup in Qatar, defeating France on penalties in a thrilling final. Lionel Messi won the Golden Ball as best player!"
  },
  {
    question: "How many players are in each squad?",
    keywords: ["players", "squad", "roster", "how many", "team size"],
    answer: "👕 Each team can register 26 players in their squad for FIFA World Cup 2026, up from 23 in previous tournaments."
  },
  {
    question: "What is the prize money?",
    keywords: ["prize", "money", "prize money", "reward", "pay", "dollars", "million"],
    answer: "💰 FIFA World Cup 2026 prize money is $1 billion total — the largest in World Cup history! The winning team receives $125 million."
  },
  {
    question: "Which teams are favorites to win?",
    keywords: ["favorite", "favourites", "best team", "win", "who will win", "strong", "predicted"],
    answer: "⭐ Top favorites include defending champions Argentina, France, Brazil, England, and Spain. But anything can happen in football!"
  },
  {
    question: "How does qualification work?",
    keywords: ["qualify", "qualification", "how to qualify", "get in", "selection"],
    answer: "🎯 Teams qualify through continental playoffs:\n🌎 CONMEBOL (South America): 6 spots\n🌍 UEFA (Europe): 16 spots\n🌍 CAF (Africa): 9 spots\n🌏 AFC (Asia): 8 spots\n🌏 CONCACAF (N/C America): 6 spots\n🌊 OFC (Oceania): 1 spot\nPlus 2 inter-confederation playoff spots!"
  },
  {
    question: "Who is the all-time top scorer in World Cup?",
    keywords: ["top scorer", "most goals", "goals", "scorer", "record", "miroslav klose"],
    answer: "⚽ Miroslav Klose of Germany holds the all-time World Cup scoring record with 16 goals across four tournaments (2002-2014)."
  },
  {
    question: "How many matches will be played?",
    keywords: ["matches", "games", "how many matches", "total games", "fixtures"],
    answer: "🔢 A total of 104 matches will be played in FIFA World Cup 2026 — up from 64 matches in Qatar 2022, due to the expanded 48-team format."
  },
  {
    question: "What time zone are the matches in?",
    keywords: ["time", "timezone", "time zone", "kick off", "kickoff", "schedule time"],
    answer: "⏰ Matches will be played across multiple time zones since the tournament spans USA, Canada, and Mexico. Match times vary by venue from ET (Eastern) to PT (Pacific)."
  },
  {
    question: "How can I buy tickets?",
    keywords: ["ticket", "tickets", "buy", "purchase", "attend", "watch live"],
    answer: "🎟️ Tickets for FIFA World Cup 2026 are sold through the official FIFA website at fifa.com. Ticket sales are done in phases through ballots and direct sales."
  }
];

// ── NLP: Tokenize & Clean ─────────────────────
function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .split(/\s+/)
    .filter(function(w) { return w.length > 1; });
}

var stopwords = ['the','is','are','was','were','a','an','in','on','at','to','for',
  'of','and','or','but','i','my','me','what','which','who','how','when',
  'where','do','does','did','can','will','would','could','should','have',
  'has','had','be','been','being','it','its','this','that','these','those'];

function removeStopwords(tokens) {
  return tokens.filter(function(t) { return stopwords.indexOf(t) === -1; });
}

function preprocess(text) {
  return removeStopwords(tokenize(text));
}

// ── TF-IDF Cosine Similarity ──────────────────
function getTermFrequency(tokens) {
  var tf = {};
  tokens.forEach(function(t) { tf[t] = (tf[t] || 0) + 1; });
  return tf;
}

function cosineSimilarity(tokensA, tokensB) {
  var tfA = getTermFrequency(tokensA);
  var tfB = getTermFrequency(tokensB);

  var allTerms = Object.keys(tfA).concat(Object.keys(tfB));
  allTerms = allTerms.filter(function(v, i, a) { return a.indexOf(v) === i; });

  var dotProduct = 0;
  var magA = 0;
  var magB = 0;

  allTerms.forEach(function(term) {
    var a = tfA[term] || 0;
    var b = tfB[term] || 0;
    dotProduct += a * b;
    magA += a * a;
    magB += b * b;
  });

  if (magA === 0 || magB === 0) return 0;
  return dotProduct / (Math.sqrt(magA) * Math.sqrt(magB));
}

function keywordBoost(userTokens, faq) {
  var boost = 0;
  faq.keywords.forEach(function(kw) {
    var kwTokens = kw.toLowerCase().split(/\s+/);
    kwTokens.forEach(function(k) {
      if (userTokens.indexOf(k) !== -1) boost += 0.15;
    });
  });
  return boost;
}

// ── Find Best FAQ Match ───────────────────────
function findBestMatch(userInput) {
  var userTokens = preprocess(userInput);
  if (userTokens.length === 0) return null;

  var bestScore = 0;
  var bestFaq = null;

  faqs.forEach(function(faq) {
    var faqTokens = preprocess(faq.question + ' ' + faq.keywords.join(' '));
    var score = cosineSimilarity(userTokens, faqTokens);
    score += keywordBoost(userTokens, faq);

    if (score > bestScore) {
      bestScore = score;
      bestFaq = faq;
    }
  });

  return bestScore > 0.08 ? { faq: bestFaq, score: bestScore } : null;
}

// ── Chat UI ───────────────────────────────────
var chatContainer = document.getElementById('chatContainer');
var userInput = document.getElementById('userInput');
var sendBtn = document.getElementById('sendBtn');

userInput.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

userInput.addEventListener('input', function() {
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
});

function sendSuggestion(text) {
  userInput.value = text;
  sendMessage();
}

function sendMessage() {
  var text = userInput.value.trim();
  if (!text) return;

  userInput.value = '';
  userInput.style.height = 'auto';
  addMessage(text, true);

  setTimeout(function() {
    var result = findBestMatch(text);
    if (result) {
      var confidence = Math.min(Math.round(result.score * 100 * 3), 99);
      addBotMessage(result.faq.answer, confidence);
    } else {
      addBotMessage("🤔 Sorry, I couldn't find a matching answer. Try asking about:\n• Host countries\n• Teams & squads\n• Schedule & venues\n• Rules & format\n• Prize money", null);
    }
  }, 600);
}

function addMessage(content, isUser) {
  var msg = document.createElement('div');
  msg.className = 'message ' + (isUser ? 'user' : 'bot');

  var avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = isUser ? '👤' : '⚽';

  var bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = content.replace(/\n/g, '<br>');

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  chatContainer.appendChild(msg);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addBotMessage(content, confidence) {
  var msg = document.createElement('div');
  msg.className = 'message bot';

  var avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = '⚽';

  var bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = content.replace(/\n/g, '<br>');

  if (confidence !== null) {
    var conf = document.createElement('div');
    conf.className = 'confidence';
    conf.textContent = '✓ Match confidence: ' + confidence + '%';
    bubble.appendChild(conf);
  }

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  chatContainer.appendChild(msg);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}
