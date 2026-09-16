(* ::Package:: *)

(* analytic_wait.wl

   GetAnalyticJ[lambda, intervals, totalTime, ECostAfter]

   Integriert E[D](s) = c0 + c1*s + c2*s^2 pro Intervall gegen die
   Exponentialverteilung in absoluter Zeit t.

   Intervall-Format (von Python uebergeben):
     intervals = {{t_start_1, t_end_1, {c0_1, c1_1, c2_1}}, ...}

   mit s = (t - wait) - t_start  (lokale Routenzeit, s in [0, dt]).

   Korrekturen:
   - Assuming[...] nur um Integrate, nicht um das Endergebnis:
     verhindert ConditionalExpression-Wrapper bei FullSimplify.
   - Kein FullSimplify am Ende: NMinimize kann Ausdruck direkt numerisch
     auswerten.
*)

BeginPackage["WarehouseOptimization`"]

GetAnalyticJ::usage =
  "GetAnalyticJ[lambda, intervals, totalTime, ECostAfter] berechnet J(wait) \
aus den E[D]-Polynomkoeffizienten pro Intervall in lokaler Zeit s. \
intervals: {{t_start, t_end, {c0, c1, c2}}, ...}";

FindCriticalLambda::usage =
  "FindCriticalLambda[intervals, totalTime, nPicksAfter, lambdaGuess] \
gibt den kritischen lambda-Wert zurueck, bei dem J'(0)=0 ist.";

Begin["`Private`"]

GetAnalyticJ[lambdaVal_?NumericQ,
             intervals_List,
             totalTimeVal_?NumericQ,
             ECostAfterVal_?NumericQ] :=

  Module[{lambda, dist, edSum, assumptions, edI0, kgDi0},

    lambda      = lambdaVal;
    dist        = ExponentialDistribution[lambda];
    assumptions = Element[Global`wait, Reals] && Global`wait >= 0;
    (* I_0: Waiting interval [0, wait] at depot. EDi0 is E[D] at route start. *)
    edI0 = If[Length[intervals] >= 1, intervals[[1, 3, 1]], 0.0];
    kgDi0 = edI0 * CDF[dist, Global`wait];

    edSum = Sum[
      With[{
        ta = intervals[[i, 1]],
        tb = intervals[[i, 2]],
        c0 = intervals[[i, 3, 1]],
        c1 = intervals[[i, 3, 2]],
        c2 = intervals[[i, 3, 3]]
      },
        (* Assuming nur fuer Integrate: stellt sicher, dass Wolfram      *)
        (* die Grenzen wait+ta und wait+tb als reell behandelt, ohne     *)
        (* einen ConditionalExpression-Wrapper im Ergebnis zu erzeugen. *)
        Assuming[assumptions,
          Integrate[
            (c0 + c1*(t - Global`wait - ta) + c2*(t - Global`wait - ta)^2)
              * PDF[dist, t],
            {t, Global`wait + ta, Global`wait + tb}
          ]
        ]
      ],
      {i, 1, Length[intervals]}
    ];

    edSum += ECostAfterVal * (1 - CDF[dist, Global`wait + totalTimeVal]);

    (* Rohes Ergebnis ohne FullSimplify zurueckgeben: NMinimize wertet  *)
    (* den Ausdruck direkt numerisch aus ohne ConditionalExpression.     *)
    Global`wait + kgDi0 + edSum
  ]


EDi0FromIntervals[intervals_List] :=
  If[Length[intervals] >= 1, intervals[[1, 3, 1]], 0.0]


(* Intervall-Format: {t_start, t_end, {c0, c1, c2}} *)
IntervalContributionC[lambda_?NumericQ, interval_List] :=
 Module[{ta, tb, dt, c0, c1, c2, dist, t},
  ta = interval[[1]];
  tb = interval[[2]];
  dt = tb - ta;
  c0 = interval[[3, 1]];
  c1 = interval[[3, 2]];
  c2 = interval[[3, 3]];
  dist = ExponentialDistribution[lambda];
  (* Wir integrieren in globaler Zeit t, lokale Zeit s = t - ta *)
  Assuming[Element[t, Reals],
    lambda * Exp[-lambda*ta] *
      Integrate[
        (c0 + c1*(t - ta) + c2*(t - ta)^2) * Exp[-lambda*(t - ta)],
        {t, ta, tb},
        Assumptions -> lambda > 0
      ]
  ]
]

TotalC[lambda_?NumericQ, intervals_List, totalTimeVal_?NumericQ, nPicksAfter_: 3] :=
 Module[{tEnd = totalTimeVal, eCostAfter},
  (* ECostAfter = n_picks_after / lambda + T_end, wie in Python *)
  eCostAfter = nPicksAfter/lambda + tEnd;
  Sum[
    IntervalContributionC[lambda, intervals[[i]]],
    {i, 1, Length[intervals]}
  ] + eCostAfter * Exp[-lambda * tEnd]
]

LambdaEquation[lambda_?NumericQ, intervals_List, totalTimeVal_?NumericQ, nPicksAfter_: 3] :=
 Module[{cRoutes, edI0},
  cRoutes = TotalC[lambda, intervals, totalTimeVal, nPicksAfter];
  edI0 = EDi0FromIntervals[intervals];
  (* For J(wait) = wait + EDi0*(1-exp(-lambda*wait)) + cRoutes*exp(-lambda*wait):
     J'(0)=0  <=>  lambda*(cRoutes - EDi0) - 1 == 0 *)
  lambda * (cRoutes - edI0) - 1
 ]

(* Findet lambda*, so dass lambda*C(lambda) == 1 *)
FindCriticalLambda[intervals_List,
                   totalTimeVal_?NumericQ,
                   nPicksAfter_: 3,
                   lambdaGuess_: 0.05] :=
 Module[{lambda},
  lambda /.
    FindRoot[
      LambdaEquation[lambda, intervals, totalTimeVal, nPicksAfter] == 0,
      {lambda, lambdaGuess},
      WorkingPrecision -> 30
    ]
]


End[]
EndPackage[]
