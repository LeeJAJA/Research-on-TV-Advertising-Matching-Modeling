#include <cstdio>
#include <cstring>
#include <queue>
#include <map>
#include <cstdlib> 
#include <set>
#include <ctime>

using namespace std;

map<int, int>ma;
queue<int>w, ww;
set<int>st;
int ml;
FILE *user, *ad, *ts;
double sigmaAll, sigmaOne[50];

const int nad = 100000;
const int firstsame = 50;
const int firstad = 15;
const int nuser = 5000;
const int del = 10;

struct u
{
	int id;
	int a[nad + 10];
	u *next;
	u()
	{
		id = 0;
		next = NULL;
		memset(a, 0, sizeof(a));
	}
}*rt, *tl;

struct uu
{
	int iduser;
	long long z;
	friend bool operator < (const uu &aa, const uu &bb)
	{
		return aa.z > bb.z;
	}
};

int type[nad+10];
map<int, u*>mmp;
priority_queue<uu>q;

struct u3
{
	int idad;
	double z;
	friend bool operator < (const u3 &aa, const u3 &bb)
	{
		if(aa.z==bb.z)
			return sigmaOne[type[aa.idad]]>sigmaOne[type[bb.idad]];
		return aa.z > bb.z;
	}
}c[nad + 10];

struct u4
{
	int id;
	int rk;
	friend bool operator < (const u4 &aa, const u4 &bb)
	{
		return aa.rk > bb.rk;
	}
}c2[500000];

priority_queue<u3>qq;

void append()
{
	u *nd = new u();
	int aa, bb, cc, dd, ee;
	fscanf(user, "%d%d%d", &aa, &bb, &cc);
	nd->id = bb;
	mmp[bb] = nd;
	fscanf(user, "%d", &cc);
	while (cc != -1)
	{
		nd->a[ma[cc]]++;
		sigmaOne[ type[ma[cc]]]+=1.0;
		sigmaAll += 1.0;
		fscanf(user, "%d", &cc);
	}
	if (rt == NULL)
	{
		rt = nd;
	}
	else
		tl->next = nd;
	tl = nd;
}

long long getSameValue(u *aa, u* bb)
{
	long long pr = 0;
	for (int i = 0;i < nad;i++)
		pr += (aa->a[i] - bb->a[i])*(aa->a[i] - bb->a[i]);
	return -pr;
}

void compare(u *aa)
{
	u *now = rt;
	while (!q.empty())
		q.pop();
	while (now != NULL)
	{
		uu newuu;
		newuu.iduser = now->id;
		newuu.z = getSameValue(now, aa);
		now = now->next;
		q.push(newuu);
		while (q.size() > firstsame)
			q.pop();
	}
}

bool g()
{
	st.clear();
	while (!qq.empty())
	{
		st.insert(qq.top().idad);
		qq.pop();
	}
	while (!ww.empty())
	{
		int p = ww.front();
		if (st.count(ww.front()) > 0)
			return true;
		ww.pop();
	}
	return false;
}



int main()
{
	long count = 0;
	user = fopen("userfinal.txt", "r");
	ad = fopen("ad5.txt", "r");
	ts = fopen("usertestfinal.txt", "r");
	for (int i = 1;i <= nad;i++)
	{
		int aa, bb, cc;
		fscanf(ad, "%d%d%d", &aa, &bb, &cc);
		ma[bb] = ++ml;
		type[ml] = cc;
	}
	for (int i = 1;i <= nuser;i++)
	{
		append();
	}
	for (int i = 1;i <= ml;i++)
		c[i].idad = i;
	double cg = 0, zs = 0;
	while (1)
	{
		++count;
		u *nd = new u();
		int aa, bb, cc, dd, ee;
		fscanf(ts, "%d%d%d", &aa, &bb, &cc);
		nd->id = bb;
		while (!w.empty()) w.pop();
		while (!ww.empty()) ww.pop();
		int bh = 0;
		fscanf(ts, "%d", &cc);
		while (cc != -1)
		{
			bh++;
			c2[bh].id = cc;
			c2[bh].rk = rand();
			fscanf(ts, "%d", &cc);
		}
		for (int i = 1;i <= bh;i++)
			if (i <= del)
				ww.push(ma[c2[i].id]);
			else
				(nd->a[ma[c2[i].id]])++;
		compare(nd);
		for (int i = 1;i <= ml;i++)
			c[i].z = 0;
		while (!q.empty())
		{
			int id = q.top().iduser;
			int pr = 0;
			for (int j = 1;j <= ml;j++)
				pr += (mmp[id]->a[j] > 0);
			u* uid = mmp[id];
			q.pop();
			for (int i = 1;i <= ml;i++)
				if (uid->a[i] > 0)
				{
					c[i].z+=1;
				}
		}
		for (int i = 1;i <= ml;i++)
		{
			qq.push(c[i]);
			while (qq.size() > firstad)
				qq.pop();
		}
		if (g())
			cg += 1.0;
		zs += 1.0;
		for (int i = 1;i <= del;i++)
			(nd->a[ma[c2[i].id]])++;
		u *p = mmp[nd->id];
		if (p == NULL)
		{
			tl->next = nd;
			tl = nd;
			mmp[nd->id] = nd;
			for (int i = 1;i <= ml;i++)
			{
				sigmaAll += nd->a[i];
				sigmaOne[type[i]] += nd->a[i];
			}
		}
		else
		{
			for (int i = 1;i <= ml;i++)
			{
				sigmaAll -= p->a[i];
				sigmaOne[type[i]] -= p->a[i];
				sigmaAll += nd->a[i];
				sigmaOne[type[i]] += nd->a[i];
				p->a[i] = nd->a[i];
			}
		}
		printf("%d: %lf\n",count , cg / zs);
	} 
}
