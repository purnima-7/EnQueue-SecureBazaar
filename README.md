# EnQueue SecureBazaar 🚀
### AI-Powered Secure Freelance Marketplace

EnQueue SecureBazaar is a full-stack freelance marketplace platform built using the MERN stack and enhanced with AI-powered face verification for secure and trustworthy user interactions.

The platform enables:
- Freelancers to create and manage gigs
- Clients to browse and purchase services
- Real-time communication between users
- Secure onboarding using biometric verification

---

# 🌟 Features

## 🔐 Secure Authentication & Face Verification
- JWT-based authentication
- Password hashing using bcryptjs
- AI-powered biometric face verification
- Prevents fake accounts and impersonation

## 💼 Gig Management
- Create, update, and delete gigs
- Browse and filter gigs dynamically
- Structured freelance service listings

## 📦 Order Management
- Complete order lifecycle tracking
- Order statuses:
  - Pending
  - In Progress
  - Delivered
  - Completed

## 💬 Real-Time Messaging
- Dedicated chat between freelancers and clients
- Persistent conversation history
- Improved project collaboration

## ⭐ Ratings & Reviews
- Review freelancers after order completion
- Dynamic rating updates
- Trust-building feedback system

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| React.js | Frontend UI |
| Node.js | Backend Runtime |
| Express.js | REST API Framework |
| MongoDB | Database |
| TensorFlow | AI Face Verification |
| OpenCV | Face Detection & Preprocessing |
| JWT | Authentication |
| bcryptjs | Password Encryption |

---

# 🧠 AI Face Verification

The platform integrates a Siamese Neural Network–based face verification system that:
- Detects faces using OpenCV
- Generates embeddings using TensorFlow
- Compares similarity between ID image and live webcam capture
- Verifies user authenticity during registration

This creates an additional trust layer beyond traditional password-based systems.

---

# 📂 Project Structure

```bash
EnQueue-SecureBazaar/
│
├── client/                 # React Frontend
├── server/                 # Node.js + Express Backend
├── ai-face-verification/   # Python AI Service
├── models/                 # MongoDB Schemas
├── routes/                 # API Routes
├── controllers/            # Backend Logic
├── middleware/             # Auth & Validation
├── public/                 # Static Assets
└── README.md
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/enqueue-securebazaar.git
cd enqueue-securebazaar
```

---

## 2️⃣ Install Dependencies

### Frontend

```bash
cd client
npm install
```

### Backend

```bash
cd ../server
npm install
```

### AI Verification Service

```bash
cd ../ai-face-verification
pip install -r requirements.txt
```

---

# ▶️ Running the Application

## Start Frontend

```bash
cd client
npm start
```

## Start Backend

```bash
cd server
npm run dev
```

## Start AI Verification Service

```bash
python app.py
```

---

# 🔑 Environment Variables

Create a `.env` file inside the server directory:

```env
MONGO_URI=your_mongodb_connection
JWT_SECRET=your_secret_key
PORT=5000
```

---

# 📸 Core Modules

- Authentication & Verification
- Gig Marketplace
- Order Management
- Messaging System
- Reviews & Ratings

---

# 🧪 Testing

The project includes:
- Unit Testing
- Integration Testing
- API Validation
- Database CRUD Testing

Major functionalities including authentication, gig creation, messaging, and order workflows were tested successfully.

---

# 🚧 Future Enhancements

- AI-based Fraud Detection
- Payment Gateway Integration
- Mobile Application
- Real-time Notifications
- ML-based Gig Recommendation System
- Multi-language Support

---

# 👨‍💻 Contributors

- Vasudev
- Purnima Singh
- Priya Gaggar

---

# 📄 License

This project is developed for academic and educational purposes.

---

# 🙌 Acknowledgements

Special thanks to **Ms. Neha** for continuous guidance, mentorship, and technical support throughout the project development.
