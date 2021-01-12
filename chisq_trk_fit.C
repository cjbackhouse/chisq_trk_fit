#include "TCanvas.h"
#include "TH2.h"

#include "Math/Factory.h"
#include "Math/Minimizer.h"

#include <map>
#include <numeric>

const int kNumWires = 500;

const double width0 = 20;

template<class T> T sqr(const T& x){return x*x;}

struct Gaus
{
  Gaus() : area(0), mu(0), sigmaSq(0) {}

  Gaus(double a, double m, double s2) : area(a), mu(m), sigmaSq(s2) {}

  double Eval(double x) const
  {
    return area/sqrt(2*M_PI*sigmaSq)*exp(-sqr(x-mu)/(2*sigmaSq));
  }

  double area;
  double mu;
  double sigmaSq;
};

struct GausSum
{
  GausSum(){}
  GausSum(const std::vector<Gaus>& es) : elem(es) {}

  std::vector<Gaus> elem;

  double Area() const
  {
    return std::accumulate(elem.begin(), elem.end(), 0.,
                           [](double s, const Gaus& g){return s+g.area;});
    /*
    double sum = 0;
    for(const Gaus& e: elem) sum += e.area;
    return sum;
    */
  }
  double Eval(double x) const
  {
    return std::accumulate(elem.begin(), elem.end(), 0.,
                           [x](double s, const Gaus& g){return s+g.Eval(x);});
    /*
    double sum = 0;
    for(const Gaus& e: elem) sum += e.Eval(x);
    return sum;
    */
  }
};

Gaus sqr(const Gaus& x)
{
  return Gaus(sqr(x.area)/sqrt(4*M_PI*x.sigmaSq),
              x.mu*x.sigmaSq,
              x.sigmaSq/2);
}

Gaus operator*(const Gaus& a, const Gaus& b)
{
  // TODO get better reference?
  // https://blog.jafma.net/2010/11/09/the-product-of-two-gaussian-pdfs-is-not-a-pdf-but-is-gaussian-a-k-a-loving-algebra/

  const double sumsigmaSq = a.sigmaSq+b.sigmaSq;

  return Gaus(Gaus(a.area*b.area, 0, sumsigmaSq).Eval(a.mu-b.mu),
              (a.mu*b.sigmaSq + b.mu*a.sigmaSq)/sumsigmaSq,
              a.sigmaSq*b.sigmaSq/sumsigmaSq);
}

GausSum operator-(const Gaus& a, const Gaus& b)
{
  GausSum ret;
  ret.elem = {a, b};
  ret.elem[1].area *= -1;
  return ret;
}

// For some reason this optimization leaves us 0.1 chisq points short of the
// true minimum compared to the naive implementation of as*as below
GausSum sqr(const GausSum& as)
{
  GausSum ret;
  ret.elem.reserve(as.elem.size() + ((as.elem.size()-1)*as.elem.size())/2);
  for(const Gaus& a: as.elem) ret.elem.push_back(sqr(a));
  for(unsigned int i = 0; i < as.elem.size(); ++i){
    for(unsigned int j = i+1; j < as.elem.size(); ++j){
      ret.elem.push_back(as.elem[i]*as.elem[j]);
      ret.elem.back().area *= 2;
    }
  }
  return ret;
}

GausSum operator*(const GausSum& as, const GausSum& bs)
{
  GausSum ret;
  ret.elem.reserve(as.elem.size() * bs.elem.size());

  for(const Gaus& a: as.elem){
    for(const Gaus& b: bs.elem){
      ret.elem.push_back(a*b);
    }
  }

  return ret;
}

GausSum operator-(const GausSum& a, const GausSum& b)
{
  GausSum ret = a;
  ret.elem.reserve(a.elem.size() + b.elem.size());
  for(const Gaus& e: b.elem){
    ret.elem.push_back(e);
    ret.elem.back().area *= -1;
  }
  return ret;
}

double chisq(const Gaus& a, const Gaus& b)
{
  return (sqr(a-b)).Area();
}

double chisq(const GausSum& as, const GausSum& bs)
{
  return (sqr(as-bs)).Area();
}

std::array<GausSum, kNumWires> gData;

void add_track(double m, double c, double x0, double x1,
               std::array<GausSum, kNumWires>& out)
{
  if(x0 > x1) std::swap(x0, x1);

  const Gaus g(1, 0, sqr(width0));

  for(int i = 0; i < kNumWires; ++i){
    if(i < x0-3*width0 || i > x1+3*width0) continue;

    Gaus hit;
    hit.sigmaSq = sqr(width0);
    hit.mu = m*i+c;
    if(i < x0){
      hit.area = g.Eval(i-x0)/g.Eval(0);
    }
    else if(i > x1){
      hit.area = g.Eval(i-x1)/g.Eval(0);
    }
    else{
      hit.area = 1;
    }

    out[i].elem.push_back(hit);
  }
}

double chisq(const std::array<GausSum, kNumWires>& a,
             const std::array<GausSum, kNumWires>& b)
{
  double ret = 0;
  for(int i = 0; i < kNumWires; ++i){
    ret += chisq(a[i], b[i]);
  }
  return ret;
}

void plot(const std::array<GausSum, kNumWires>& data)
{
  TH2F* hdata = new TH2F("", "", kNumWires, 0, kNumWires, 500, 0, 500);

  for(unsigned int x = 0; x < kNumWires; ++x){
    for(int iy = 0; iy < hdata->GetNbinsY()+2; ++iy){
      const double y = hdata->GetYaxis()->GetBinCenter(iy);
      hdata->Fill(x, y, data[x].Eval(y));
    }
  }

  hdata->SetMinimum(-1e-10);
  hdata->SetMaximum(.025);
  hdata->Draw("colz");
}

class Func: public ROOT::Math::IMultiGenFunction // BaseFunctionMultiDim
{
public:
  virtual Func* Clone() const override {abort();}

  virtual double DoEval(const double* pars) const override
  {
    const double m1 = pars[0];
    const double c1 = pars[1];
    const double x01 = pars[2];
    const double x11 = pars[3];

    const double m2 = pars[4];
    const double c2 = pars[5];
    const double x02 = pars[6];
    const double x12 = pars[7];

    const double m3 = pars[8];
    const double c3 = pars[9];
    const double x03 = pars[10];
    const double x13 = pars[11];

    std::array<GausSum, kNumWires> pred;
    add_track(m1, c1, x01, x11, pred);
    add_track(m2, c2, x02, x12, pred);
    add_track(m3, c3, x03, x13, pred);

    const double score = chisq(pred, gData);

    std::cout << m1 << " " << c1 << " " << m2 << " " << c2 << " " << score << std::endl;

    static double bestScore = 1e10;

    if(score < bestScore*.95){
      bestScore = score;
      plot(pred);
      gPad->Update();
    }

    return score;
  }

  virtual unsigned int NDim() const override {return 2;}
} func;

void chisq_trk_fit()
{
  add_track(.37, 100, 100, 400, gData);
  add_track(.9, 100, 50, 300, gData);
  add_track(-.5, 150, 100, 150, gData);

  new TCanvas;
  plot(gData);
  gPad->Update();
  new TCanvas;

  ROOT::Math::Minimizer* mnMin = ROOT::Math::Factory::CreateMinimizer("Minuit2", "Combined");

  mnMin->SetVariable(0, "m1", 0, 1);
  mnMin->SetVariable(1, "c1", 200, 50);
  mnMin->SetVariable(2, "x01", 20, 50);
  mnMin->SetVariable(3, "x11", 480, 50);
  mnMin->SetVariable(4, "m2", -1/*.9*/, 1);
  mnMin->SetVariable(5, "c2", 400/*100*/, 50);
  mnMin->SetVariable(6, "x02", 20, 50);
  mnMin->SetVariable(7, "x12", 480, 50);

  mnMin->SetVariable(8, "m3", -.5, 1);
  mnMin->SetVariable(9, "c3", 300, 50);
  mnMin->SetVariable(10, "x03", 20, 50);
  mnMin->SetVariable(11, "x13", 480, 50);

  mnMin->SetFunction(func);

  mnMin->Minimize();

  std::cout << "Done " << mnMin->MinValue() << std::endl;

  /*
  const double m = mnMin->X()[0];
  const double c = mnMin->X()[1];

  std::array<GausSum, kNumWires> pred;
  add_track(m, c, 0, 500, pred);

  new TCanvas;
  plot(pred);
  */
}
