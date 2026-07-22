\section{Introduction}

Significant progress in robot manipulation, perception, and
planning is bringing autonomous robots closer to assisting people in their homes~\cite{brohan2023rt, padalkar2023open,
jenamani2024flair, jenamani2024bitetransfer, nanavati2025lessons}.
Yet most of what we know about how such systems perform comes from short-term evaluations --- single sessions or multi-day studies in controlled settings~\cite{jenamani2024bitetransfer,
jenamani2025feast, nanavati2025lessons}. These reveal whether a
system works. They cannot reveal what happens over weeks of real
use, when novelty fades, contexts diversify, and the relationship between a user and a robot begins to evolve.

This evolution is not one-sided. The robot must track preferences that shift with context, mood, and fatigue --- and unlike short-term settings, long-term deployment exposes it to the full diversity of a user's life, demanding efficient personalization across many contexts with minimal burden on the user. At the same time, users accumulate experience: they learn the robot's capabilities and failure modes, discover when to intervene, and adapt their behavior and environment accordingly. We call this bidirectional process \emph{co-adaptation}, and aim to study this defining feature of long-term human-robot deployment in our paper.

In this work, we study co-adaptation through a month-long in-home deployment of a robot mealtime-assistance system with a care recipient with Multiple Sclerosis, conducted using \emph{community-based participatory research} (CBPR)~\cite{hacker2013community}. Prior work estimates that novelty effects in human-robot interaction persist for around 21 days~\cite{bajones2019results} --- our deployment crosses this threshold, allowing us to observe co-adaptation beyond initial adjustment and into sustained use. Our system operates on a mobile manipulator and autonomously executes the full mealtime pipeline: retrieving a meal from the refrigerator, heating it in the microwave, placing it on the table, assisting the user with feeding, and returning the plate to the sink (Figure~\ref{fig:teaser}). A single pipeline execution can take over an hour --- creating a setting in which preferences surface across diverse subtasks, failures accumulate over time, and co-adaptation is given the time and complexity it needs to unfold. To support the robot's side of adaptation, we propose a preference learning framework grounded in the hypothesis that user preferences form correlated bundles driven by latent state --- such that a single correction from the user for one preference dimension carries signal across multiple preference dimensions simultaneously. 

% We use a large language model for joint prediction across preference dimensions and evaluate this against baselines including multi-arm bandits and independent per-dimension learning.

% Statement of contributions
Overall, our contributions include:
\begin{itemize}
    \item \textbf{An end-to-end mealtime-assistance system} on a mobile manipulator that autonomously executes the full mealtime pipeline --- from retrieving food from the refrigerator to returning the plate to the sink --- deployed in a real home.
    \item \textbf{A preference learning framework} grounded in the hypothesis that user preferences form correlated bundles driven by latent state, enabling efficient long-term personalization via LLM-based joint prediction across preference dimensions.
    \item \textbf{A month-long in-home deployment} with a care
    recipient with mobility limitations, quantifying system
    performance, personalization efficiency, and the dynamics of
    co-adaptation over sustained use.
    \item \textbf{Lessons learned} from deployment, characterizing how users and robots mutually adapt over time and surfacing design principles for long-term home robot systems.
\end{itemize}
