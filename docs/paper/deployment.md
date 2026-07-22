\section{Deployment}
\label{sec:deployment}

\subsection{Participant}

We conduct this study using \emph{community-based participatory research} (CBPR)~\cite{hacker2013community} in collaboration with a community researcher (CR) who is a co-author on this paper. This approach, involving co-design and in-depth evaluation with one or two CRs, is well-established in assistive technology research~\cite{kushalnagar2020teleconference, haidenhofer2024research, fussenegger2022depending} and increasingly common in assistive robotics~\cite{chen2012robots, moharana2019robots, nanavati2023design, padmanabha2024independence, ranganeni2024robots, nanavati2025lessons}.

Our CR is a female with Multiple Sclerosis who retains partial arm and neck mobility. She interacts with the robot's web interface using her left arm. She prefers to lean forward to receive bites. The CR is a co-author on this paper and has been involved throughout the system's design and evaluation.

\subsection{Study Procedure}

\textbf{Preparation Phase.}
Prior to the deployment, we spend five days at the CR's home preparing the system for sustained use. This preparation serves two purposes. First, we robustify our system's skills to the specific conditions of her environment, including lighting, furniture placement, and table height, which differ from our lab setting. Second, we conduct a series of trial meals with the CR to familiarize her with the full mealtime pipeline and to establish her baseline preferences (e.g., feeding side, robot speed, interaction modality). These trial meals also allow us to identify and resolve any accessibility issues with the web interface given her range of motion. The preparation phase concludes once both the CR and the research team are satisfied that the system is stable and ready for unsupervised daily use.

\textbf{Deployment Phase.}
The deployment spans four continuous weeks, with one meal assisted per day on weekdays, subject to the CR's availability. Each meal consists of the full mealtime pipeline: retrieving a meal from the refrigerator, heating it in the microwave, placing it on the table, assisting the CR with feeding, and returning the plate to the sink. A researcher is present for monitoring purposes but does not intervene unless safety requires it.

\subsection{Measures}
\label{sec:measures}

\textbf{Objective Metrics.}
We track three categories of objective metrics across all meals.

\emph{Preference Learning Efficiency.} We log the total number of corrections the CR provides before the robot converges to her preferences. Since preference dimensions may be correlated, a single correction can carry signal across multiple dimensions simultaneously; tracking total corrections captures the overall adaptation burden on the CR.

\emph{Interventions and Explanations.} We log researcher interventions per meal, categorized into hardware, software, skills, and safety. We also record the number of questions the CR directs to the research team rather than to the system.

\emph{Skill Success Rates.} We measure success rates for each autonomous skill that relies on perception: fridge door opening, microwave door opening and closing, bite acquisition, and bite transfer. Fixed-motion actions always succeed and are excluded. Reasons for individual skill failures are logged per meal and reported in the Appendix.

\textbf{Daily Measures.}
After each meal, we ask the CR three questions (full text in Appendix). The first two use a 5-point Likert scale (1 = Strongly Disagree, 5 = Strongly Agree):
\begin{itemize}
    \item \textbf{Safety}: How safe did you feel during today's meal?
    \item \textbf{Satisfaction}: How satisfied were you with the robot's performance during today's meal?
\end{itemize}
The third is open-ended:
\begin{itemize}
    \item \textbf{Adaptation}: What, if anything, did you learn about the robot today, or how did your interaction with it change?
\end{itemize}

\textbf{End-of-Study Measures.}
At the end of the deployment, we administer the following on a 5-point Likert scale (1 = Strongly Disagree, 5 = Strongly Agree).

\emph{Technology Acceptance Model (TAM).} We administer five items~\cite{davis1989user}:
\begin{itemize}
    \item \textbf{Perceived Usefulness}: Using this meal-assistance system will make me more independent in eating.
    \item \textbf{Perceived Ease of Use}: This meal-assistance system is easy to use.
    \item \textbf{Attitude Toward Using}: Using the meal-assistance system for improving my independence is a good idea.
    \item \textbf{Behavioral Intention to Use}: Assuming I have access to this meal-assistance system, I predict that I would use it in my daily life.
    \item \textbf{Perceived Enjoyment}: I find using this meal-assistance system to be enjoyable.
\end{itemize}

\emph{Control and Independence.} We administer two items twice --- once for the robot and once for the human caregiver --- to enable direct comparison:
\begin{itemize}
    \item \textbf{Control}: I feel in control of my feeding experience when assisted by my caregiver / the robot.
    \item \textbf{Independence}: I feel a sense of independence when I receive assistance from my caregiver / the robot.
\end{itemize}

\emph{Co-Adaptation.} We include two items to directly assess the bidirectional adaptation central to this work:
\begin{itemize}
    \item \textbf{Robot Adaptation}: The robot got better at understanding my preferences over the course of the study.
    \item \textbf{User Adaptation}: I got better at working with the robot over the course of the study.
\end{itemize}
