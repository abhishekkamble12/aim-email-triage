import random
from typing import List, Tuple
from .models import EmailFull, EmailCategory, PriorityLevel, RouteOption, EmailGroundTruth

class EmailGenerator:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        
        self.templates = [
            # URGENT (5)
            {"category": EmailCategory.urgent, "subject": "Server Outage Down", "body": "Critical failure in prod database. Immediate action required.", "priority": PriorityLevel.critical, "route": RouteOption.escalate},
            {"category": EmailCategory.urgent, "subject": "Security Breach", "body": "Unauthorized access detected on main server.", "priority": PriorityLevel.critical, "route": RouteOption.escalate},
            {"category": EmailCategory.urgent, "subject": "Immediate Invoice Overdue", "body": "Your account will be suspended in 1 hour if not paid.", "priority": PriorityLevel.high, "route": RouteOption.escalate},
            {"category": EmailCategory.urgent, "subject": "Data Loss Detected", "body": "We have detected corruption in the latest backup.", "priority": PriorityLevel.critical, "route": RouteOption.escalate},
            {"category": EmailCategory.urgent, "subject": "Urgent API Deprecation", "body": "The v1 API is shutting down in 24 hours.", "priority": PriorityLevel.high, "route": RouteOption.inbox},

            # NORMAL (5)
            {"category": EmailCategory.normal, "subject": "Quarterly Review Meeting", "body": "Please find attached the schedule for our quarterly review.", "priority": PriorityLevel.medium, "route": RouteOption.inbox},
            {"category": EmailCategory.normal, "subject": "Project Status Update", "body": "The latest branch has been merged into main.", "priority": PriorityLevel.low, "route": RouteOption.inbox},
            {"category": EmailCategory.normal, "subject": "Expense Report Approved", "body": "Your travel expenses have been processed.", "priority": PriorityLevel.medium, "route": RouteOption.archive},
            {"category": EmailCategory.normal, "subject": "Hardware Request", "body": "Please approve the request for the new testing phone.", "priority": PriorityLevel.medium, "route": RouteOption.review},
            {"category": EmailCategory.normal, "subject": "Welcome New Employee", "body": "Let's welcome John to the engineering team.", "priority": PriorityLevel.low, "route": RouteOption.inbox},

            # SPAM (5)
            {"category": EmailCategory.spam, "subject": "You Won A Tesla!", "body": "Click here to claim your free electric car.", "priority": PriorityLevel.low, "route": RouteOption.trash},
            {"category": EmailCategory.spam, "subject": "Cheap prescriptions online", "body": "Buy cheap meds without prescription now.", "priority": PriorityLevel.low, "route": RouteOption.trash},
            {"category": EmailCategory.spam, "subject": "Enlarge your business today", "body": "Our marketing software will 10x your revenue.", "priority": PriorityLevel.low, "route": RouteOption.trash},
            {"category": EmailCategory.spam, "subject": "I have your passwords", "body": "Send bitcoin or I will release your browsing history.", "priority": PriorityLevel.high, "route": RouteOption.trash},
            {"category": EmailCategory.spam, "subject": "Hot singles in your area", "body": "Meet people near you tonight. Click here.", "priority": PriorityLevel.low, "route": RouteOption.trash},

            # PROMOTIONS (5)
            {"category": EmailCategory.promotions, "subject": "50% off holiday sale", "body": "Use code HOLIDAY50 for half off your entire cart.", "priority": PriorityLevel.low, "route": RouteOption.archive},
            {"category": EmailCategory.promotions, "subject": "Last chance for early bird tickets", "body": "Reserve your spot at the conference before prices go up.", "priority": PriorityLevel.medium, "route": RouteOption.inbox},
            {"category": EmailCategory.promotions, "subject": "Buy one get one free", "body": "Don't miss our weekend BOGO promotion.", "priority": PriorityLevel.low, "route": RouteOption.archive},
            {"category": EmailCategory.promotions, "subject": "New features released", "body": "Check out the newly added dashboard widgets.", "priority": PriorityLevel.low, "route": RouteOption.archive},
            {"category": EmailCategory.promotions, "subject": "Your free trial is ending", "body": "Upgrade now to keep your premium benefits.", "priority": PriorityLevel.medium, "route": RouteOption.review},

            # SOCIAL (5)
            {"category": EmailCategory.social, "subject": "Mark liked your post", "body": "Mark commented on your recent photo upload.", "priority": PriorityLevel.low, "route": RouteOption.archive},
            {"category": EmailCategory.social, "subject": "3 new connection requests", "body": "You have people waiting to connect with you.", "priority": PriorityLevel.low, "route": RouteOption.inbox},
            {"category": EmailCategory.social, "subject": "Jane mentioned you", "body": "Jane tagged you in a recent discussion thread.", "priority": PriorityLevel.medium, "route": RouteOption.inbox},
            {"category": EmailCategory.social, "subject": "Trending in your network", "body": "See what your connections are talking about today.", "priority": PriorityLevel.low, "route": RouteOption.archive},
            {"category": EmailCategory.social, "subject": "Weekend plan discussion", "body": "Are we still meeting up for coffee on Saturday?", "priority": PriorityLevel.medium, "route": RouteOption.inbox},

            # UPDATES (5)
            {"category": EmailCategory.updates, "subject": "Shipping Confirmation", "body": "Your order #12345 has been shipped via UPS.", "priority": PriorityLevel.low, "route": RouteOption.archive},
            {"category": EmailCategory.updates, "subject": "Password reset successful", "body": "Your account password was recently changed.", "priority": PriorityLevel.high, "route": RouteOption.review},
            {"category": EmailCategory.updates, "subject": "Action required: Terms of Service", "body": "Please accept the updated terms of service.", "priority": PriorityLevel.high, "route": RouteOption.inbox},
            {"category": EmailCategory.updates, "subject": "Weekly summary report", "body": "Here is your activity summary for the past week.", "priority": PriorityLevel.low, "route": RouteOption.archive},
            {"category": EmailCategory.updates, "subject": "System downtime scheduled", "body": "Maintenance will occur this weekend from 2am to 4am.", "priority": PriorityLevel.medium, "route": RouteOption.inbox},

            # FORUMS (5)
            {"category": EmailCategory.forums, "subject": "Daily digest: Python Developers", "body": "Top 10 new discussions in the Python community.", "priority": PriorityLevel.low, "route": RouteOption.archive},
            {"category": EmailCategory.forums, "subject": "New reply in: How to fix memory leak", "body": "User 'dev123' replied to your question.", "priority": PriorityLevel.medium, "route": RouteOption.inbox},
            {"category": EmailCategory.forums, "subject": "Moderator alert: Rule violation", "body": "A post in your thread was flagged for review.", "priority": PriorityLevel.medium, "route": RouteOption.review},
            {"category": EmailCategory.forums, "subject": "Weekly newsletter: Machine Learning", "body": "Read the latest papers and tutorial links.", "priority": PriorityLevel.low, "route": RouteOption.archive},
            {"category": EmailCategory.forums, "subject": "Welcome to the community!", "body": "Please read our guidelines before posting.", "priority": PriorityLevel.low, "route": RouteOption.archive},
        ]
        
        self.phishing_templates = [
            # 4 PHISHING
            {"category": EmailCategory.spam, "subject": "URGENT: Verify your bank account", "body": "Your account will be locked. Click this link: http://bit.ly/fake-bank", "priority": PriorityLevel.critical, "route": RouteOption.trash, "is_phishing": True},
            {"category": EmailCategory.spam, "subject": "Invoice Paid Automatically", "body": "We charged you $400. If this is wrong, click to refund: http://shady-refund.bz", "priority": PriorityLevel.high, "route": RouteOption.trash, "is_phishing": True},
            {"category": EmailCategory.spam, "subject": "IT Helpdesk: Reset your Domain Password", "body": "Your corporate password expires today. Reset it here: http://corp-reset-portal.cn", "priority": PriorityLevel.critical, "route": RouteOption.trash, "is_phishing": True},
            {"category": EmailCategory.spam, "subject": "Package Delivery Failed", "body": "We missed you! Reschedule delivery by paying $1 fee: http://post-fee.ru", "priority": PriorityLevel.medium, "route": RouteOption.trash, "is_phishing": True},
        ]

    def generate_emails(self, num: int, has_phishing: bool = False, ambiguity_level: float = 0.0) -> Tuple[List[EmailFull], List[EmailGroundTruth]]:
        emails = []
        truths = []
        
        # Decide how many are phishing
        num_phishing = 0
        if has_phishing:
            # Randomly 0 to 20% of emails might be phishing
            num_phishing = min(num, self.rng.randint(0, max(1, int(num * 0.2))))
        
        num_normal = num - num_phishing

        for _ in range(num_normal):
            template = self.rng.choice(self.templates)
            self._build_email(template, False, ambiguity_level, emails, truths)

        for _ in range(num_phishing):
            template = self.rng.choice(self.phishing_templates)
            self._build_email(template, True, ambiguity_level, emails, truths)

        # Shuffle the lists together so phishing isn't all at the end
        combined = list(zip(emails, truths))
        self.rng.shuffle(combined)
        
        emails_shuffled = []
        truths_shuffled = []
        for e, t in combined:
            emails_shuffled.append(e)
            truths_shuffled.append(t)

        return emails_shuffled, truths_shuffled

    def _build_email(self, template, is_phishing, ambiguity_level, emails, truths):
        eid = str(self.rng.randint(10000, 99999))
        sender = f"user_{self.rng.randint(1,999)}@example.com"
        subject = template["subject"]
        body = template["body"]
        
        if self.rng.random() < ambiguity_level:
            body += " ... (context missing)"
            subject = "Re: " + subject
            
        full = EmailFull(
            id=eid,
            subject=subject,
            sender=sender,
            preview=body[:50],
            body=body
        )
        
        truth = EmailGroundTruth(
            id=eid,
            is_phishing=is_phishing,
            true_category=template["category"],
            true_priority=template["priority"],
            true_route=template["route"]
        )
        
        emails.append(full)
        truths.append(truth)
