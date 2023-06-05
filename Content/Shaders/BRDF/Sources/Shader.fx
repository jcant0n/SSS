[Begin_ResourceLayout]

	cbuffer PerDrawCall : register(b0)
	{
		float4x4 WorldViewProj	: packoffset(c0);	[WorldViewProjection]
		float4x4 World			: packoffset(c4); 	[World]
	};

	cbuffer PerCamera : register(b1)
	{
		float3 CameraPosition		: packoffset(c0.x); [CameraPosition]
	}
	
	cbuffer Parameters : register(b2)
	{
		float3 LightPosition	: packoffset(c0.x); [Default(0,0,1)]
		float ambientFactor 	: packoffset(c0.w); [Default(0.02)]
		float Metallic			: packoffset(c1.x);

		float Reflectance		: packoffset(c1.z); [Default(0.3)]
		float irradiPerp 		: packoffset(c1.w); [Default(10)]
	};

	Texture2D BaseTexture				: register(t0);
	Texture2D RoughnessAOThickness		: register(t1);
	Texture2D NormalTexture				: register(t2);
	TextureCube IBLIrradianceTexture	: register(t3); [IBLIrradiance]
	SamplerState TextureSampler			: register(s0);
	
[End_ResourceLayout]

[Begin_Pass:Default]
	[Profile 10_0]
	[Entrypoints VS=VS PS=PS]

	#define PI 3.14159265359f
	#define RECIPROCAL_PI2 0.15915494f
	
	struct VS_IN
	{
		float4 position : POSITION;
		float3 normal	: NORMAL;
		float4 tangent	: TANGENT;
		float2 texCoord : TEXCOORD;
	};

	struct PS_IN
	{
		float4 position 	: SV_POSITION;
		float3 normal		: NORMAL0;
		float3 tangent		: TANGENT0;
		float3 bitangent	: BINORMAL0;
		float2 texCoord 	: TEXCOORD0;
		float3 positionWS 	: TEXCOORD1;
	};

	struct Surface
	{
		float3 albedo;
		float AO;
		float3 position;
		float thinkness;
		float3 normal;
		float reflectance;
		float3 viewVector;
		float metallic;
		float3 reflectVector;
		float roughness;
		float NdotV;
		
		
		inline void Create( in float3 color,
							in float3 P,
							in float3 N,	
							in float3 viewPos,
							in float sAO,
							in float sthinkness,
							in float sroughness,
							in float smetallic,
							in float sreflectance)
		{
			albedo = color;
			position = P;
			thinkness = sthinkness;
			normal = N;
			viewVector = normalize(viewPos - P);
			reflectVector = normalize(reflect(viewPos, N));
			NdotV = saturate(dot(N, viewPos));
			AO = sAO;
			thinkness = sthinkness;
			metallic = smetallic;
			roughness = sroughness * sroughness;
			reflectance = sreflectance;
		}
	};
	
	float3 fresnelSchlick(float cosTheta, float3 F0)
	{
  		return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
	} 
	
	struct SurfaceToLight
	{
		float3 lightVector;
		float NdotL;
		float3 halfVector;
		float NdotH;
		float3 fresnelTerm;
		float VdotH;
		
		inline void Create(in Surface surface, in float3 lightDir)
		{
			lightVector = lightDir;
			halfVector = normalize(lightDir + surface.viewVector);
			NdotL = saturate(dot(lightVector, surface.normal));
			NdotH = saturate(dot(surface.normal, halfVector));
			VdotH = saturate(dot(surface.viewVector, halfVector));
			
			float3 f0 = 0.16 * (surface.reflectance * surface.reflectance);
			f0 = lerp(f0, surface.albedo, surface.metallic);
			fresnelTerm = fresnelSchlick(VdotH, f0);
		}
	};

	PS_IN VS(VS_IN input)
	{
		PS_IN output = (PS_IN)0;

		output.position = mul(input.position, WorldViewProj);
		
		output.positionWS = mul(input.position, World).xyz;
		output.normal = mul(float4(input.normal, 0), World).xyz;
		output.tangent = mul(input.tangent, World).xyz;
		output.bitangent = cross(output.normal, output.tangent) * input.tangent.w;
		
		output.texCoord = input.texCoord;

		return output;
	}

	float3 GammaToLinear(in float3 color)
	{
		return pow(color, 2.2);
	}
	
	float D_GGX(float NoH, float roughness)
	{
		float alpha = roughness * roughness;
		float alpha2 = alpha * alpha;
		float NoH2 = NoH * NoH;
		float b = (NoH2 * (alpha2 - 1.0) + 1.0);
		return alpha2 / (PI * b * b);
	}
	
	float G1_GGX_Schlick(float NdotV, float roughness)
	{
		//float r = roughness; // original
		float r = 0.5 + 0.5 * roughness; // Disney remapping
		float k = (r * r) / 2.0;
		float denom = NdotV * (1.0 - k) + k;
		return NdotV / denom;
	}
	
	float G_Smith(float NoV, float NoL, float roughness) 
	{
		float g1_l = G1_GGX_Schlick(NoL, roughness);
		float g1_v = G1_GGX_Schlick(NoV, roughness);
		return g1_l * g1_v;
	}
	
	float3 BRDFSpecular(in Surface surface, in SurfaceToLight surface2light)
	{
		float3 F = surface2light.fresnelTerm;
		float D = D_GGX(surface2light.NdotH, surface.roughness);
		float G = G_Smith(surface.NdotV, surface2light.NdotL, surface.roughness);
		
		return (D * G * F) / max(4.0 * surface.NdotV * surface2light.NdotL, 0.001);
	}
	
	float3 BRDFDiffuse(in Surface surface, in SurfaceToLight surface2light)
	{
		float3 diffuseIrradiance = IBLIrradianceTexture.Sample(TextureSampler, surface.normal).rgb;
	
		float3 rhoD = 1.0 - surface2light.fresnelTerm; // if not specular, use as diffuse
		rhoD *= 1.0 - surface.metallic; // no diffuse for metals
		
		
		float3 ambient = surface.albedo * surface.AO * ambientFactor * diffuseIrradiance;
		float3 diffuse = rhoD * surface.albedo / PI;
		float irradiance = max(dot(surface2light.lightVector, surface.normal), 0.0) * irradiPerp;
		
		return ambient + diffuse * irradiance;
	}

	float4 PS(PS_IN input) : SV_Target
	{
		float3 base = GammaToLinear(BaseTexture.Sample(TextureSampler, input.texCoord).rgb);		
		float3 RAT = RoughnessAOThickness.Sample(TextureSampler, input.texCoord).xyz;
				
		float3 normalTex = NormalTexture.Sample(TextureSampler, input.texCoord).rgb * 2 - 1;
		float3x3 tangentToWorld = float3x3(normalize(input.tangent), normalize(input.bitangent), normalize(input.normal));
		float3 normal = normalize(mul(normalTex, tangentToWorld));
		
		Surface surface;
		surface.Create(base,input.positionWS, normal, CameraPosition, RAT.y, RAT.z, RAT.x, Metallic, Reflectance);
		
		float3 lightDir = normalize(LightPosition - input.positionWS);
		SurfaceToLight surface2light;
		surface2light.Create(surface, lightDir);
		
		float3 diffuse = BRDFDiffuse(surface, surface2light);
		float3 specular = BRDFSpecular(surface, surface2light);
		
		return float4(diffuse + specular, 1.0);
	}

[End_Pass]