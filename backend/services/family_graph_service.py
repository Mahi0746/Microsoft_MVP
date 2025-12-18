# HealthSync AI - Family Health Graph Service
import asyncio
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import structlog
import numpy as np
from collections import defaultdict, deque

from config import settings
from services.db_service import DatabaseService


logger = structlog.get_logger(__name__)


class FamilyGraphService:
    """Service for managing family health graphs and inherited risk calculations."""
    
    # Genetic relationship coefficients (proportion of shared genes)
    RELATIONSHIP_COEFFICIENTS = {
        "father": 0.5,
        "mother": 0.5,
        "brother": 0.5,
        "sister": 0.5,
        "son": 0.5,
        "daughter": 0.5,
        "grandfather_paternal": 0.25,
        "grandmother_paternal": 0.25,
        "grandfather_maternal": 0.25,
        "grandmother_maternal": 0.25,
        "uncle_paternal": 0.25,
        "aunt_paternal": 0.25,
        "uncle_maternal": 0.25,
        "aunt_maternal": 0.25,
        "cousin": 0.125,
        "spouse": 0.0  # No genetic relationship
    }
    
    # Disease heritability estimates (how much genetics contribute)
    DISEASE_HERITABILITY = {
        "diabetes": 0.72,
        "heart_disease": 0.57,
        "hypertension": 0.62,
        "cancer": 0.33,
        "stroke": 0.38,
        "alzheimers": 0.79,
        "depression": 0.40,
        "osteoporosis": 0.85,
        "asthma": 0.75,
        "obesity": 0.70
    }
    
    # Population prevalence rates (baseline risk)
    POPULATION_PREVALENCE = {
        "diabetes": 0.11,
        "heart_disease": 0.065,
        "hypertension": 0.45,
        "cancer": 0.38,
        "stroke": 0.025,
        "alzheimers": 0.067,
        "depression": 0.084,
        "osteoporosis": 0.16,
        "asthma": 0.083,
        "obesity": 0.36
    }
    
    @classmethod
    async def create_family_graph(cls, user_id: str) -> Dict[str, Any]:
        """Create initial family health graph for user."""
        
        try:
            # Check if graph already exists
            existing_graph = await DatabaseService.mongodb_find_one(
                "family_graph",
                {"user_id": user_id}
            )
            
            if existing_graph:
                return existing_graph
            
            # Create new family graph
            family_graph = {
                "user_id": user_id,
                "family_members": [],
                "inherited_risks": {},
                "risk_calculations": {
                    "algorithm_version": "v2.1.0",
                    "calculation_date": datetime.utcnow(),
                    "factors_considered": [],
                    "confidence_scores": {}
                },
                "family_tree_visualization": {
                    "nodes": [{"id": "user", "name": "You", "generation": 0, "x": 400, "y": 300}],
                    "links": [],
                    "layout_config": {
                        "width": 800,
                        "height": 600,
                        "node_radius": 30,
                        "link_distance": 100
                    }
                },
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Insert into MongoDB
            await DatabaseService.mongodb_insert_one("family_graph", family_graph)
            
            logger.info("Family graph created", user_id=user_id)
            
            return family_graph
            
        except Exception as e:
            logger.error("Failed to create family graph", user_id=user_id, error=str(e))
            raise
    
    @classmethod
    async def add_family_member(
        cls,
        user_id: str,
        member_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add a family member to the health graph."""
        
        try:
            # Get existing family graph
            family_graph = await DatabaseService.mongodb_find_one(
                "family_graph",
                {"user_id": user_id}
            )
            
            if not family_graph:
                family_graph = await cls.create_family_graph(user_id)
            
            # Validate relationship
            relation = member_data.get("relation")
            if relation not in cls.RELATIONSHIP_COEFFICIENTS:
                raise ValueError(f"Invalid relationship: {relation}")
            
            # Create family member object
            member_id = f"fm_{len(family_graph['family_members']) + 1}_{relation}"
            
            family_member = {
                "id": member_id,
                "relation": relation,
                "name": member_data.get("name"),
                "birth_year": member_data.get("birth_year"),
                "death_year": member_data.get("death_year"),
                "health_conditions": member_data.get("health_conditions", []),
                "age_of_onset": member_data.get("age_of_onset", {}),
                "genetic_markers": member_data.get("genetic_markers", {}),
                "lifestyle_factors": member_data.get("lifestyle_factors", {}),
                "added_date": datetime.utcnow()
            }
            
            # Add to family members list
            family_graph["family_members"].append(family_member)
            
            # Recalculate inherited risks
            family_graph["inherited_risks"] = await cls._calculate_inherited_risks(
                family_graph["family_members"]
            )
            
            # Update visualization
            family_graph["family_tree_visualization"] = cls._generate_tree_visualization(
                family_graph["family_members"]
            )
            
            # Update metadata
            family_graph["updated_at"] = datetime.utcnow()
            family_graph["risk_calculations"]["calculation_date"] = datetime.utcnow()
            
            # Save to database
            await DatabaseService.mongodb_update_one(
                "family_graph",
                {"user_id": user_id},
                {"$set": family_graph}
            )
            
            logger.info(
                "Family member added",
                user_id=user_id,
                relation=relation,
                conditions=len(family_member["health_conditions"])
            )
            
            return {
                "member_id": member_id,
                "updated_risks": family_graph["inherited_risks"],
                "message": "Family member added successfully"
            }
            
        except Exception as e:
            logger.error("Failed to add family member", user_id=user_id, error=str(e))
            raise
    
    @classmethod
    async def _calculate_inherited_risks(cls, family_members: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate inherited disease risks using genetic algorithms."""
        
        try:
            # Collect all diseases mentioned in family
            all_diseases = set()
            for member in family_members:
                all_diseases.update(member.get("health_conditions", []))
            
            # Add common diseases even if not in family history
            all_diseases.update(cls.DISEASE_HERITABILITY.keys())
            
            inherited_risks = {}
            
            for disease in all_diseases:
                risk = await cls._calculate_disease_risk(disease, family_members)
                inherited_risks[disease] = risk
            
            return inherited_risks
            
        except Exception as e:
            logger.error("Risk calculation failed", error=str(e))
            return {}
    
    @classmethod
    async def _calculate_disease_risk(cls, disease: str, family_members: List[Dict[str, Any]]) -> float:
        """Calculate inherited risk for a specific disease using Bayesian approach."""
        
        try:
            # Get baseline population risk
            baseline_risk = cls.POPULATION_PREVALENCE.get(disease, 0.1)
            
            # Get heritability factor
            heritability = cls.DISEASE_HERITABILITY.get(disease, 0.5)
            
            # Calculate genetic risk contribution
            genetic_risk = 0.0
            total_genetic_weight = 0.0
            
            for member in family_members:
                if disease in member.get("health_conditions", []):
                    # Get relationship coefficient
                    relation = member.get("relation", "")
                    genetic_coefficient = cls.RELATIONSHIP_COEFFICIENTS.get(relation, 0.0)
                    
                    if genetic_coefficient > 0:
                        # Age of onset affects risk (earlier onset = higher genetic component)
                        age_of_onset = member.get("age_of_onset", {}).get(disease)
                        age_factor = cls._calculate_age_factor(age_of_onset, disease)
                        
                        # Lifestyle factors can modify genetic risk
                        lifestyle_factor = cls._calculate_lifestyle_factor(
                            member.get("lifestyle_factors", {}), disease
                        )
                        
                        # Calculate weighted genetic contribution
                        member_risk = genetic_coefficient * age_factor * lifestyle_factor
                        genetic_risk += member_risk
                        total_genetic_weight += genetic_coefficient
            
            # Normalize genetic risk
            if total_genetic_weight > 0:
                genetic_risk = genetic_risk / total_genetic_weight
            
            # Combine baseline risk with genetic risk using heritability
            # Risk = baseline + (genetic_component * heritability)
            final_risk = baseline_risk + (genetic_risk * heritability)
            
            # Apply family clustering effect (multiple affected relatives)
            affected_count = sum(
                1 for member in family_members 
                if disease in member.get("health_conditions", [])
            )
            
            if affected_count > 1:
                clustering_factor = min(1.5, 1.0 + (affected_count - 1) * 0.2)
                final_risk *= clustering_factor
            
            # Cap risk at reasonable maximum
            final_risk = min(0.85, max(baseline_risk, final_risk))
            
            return float(final_risk)
            
        except Exception as e:
            logger.error("Disease risk calculation failed", disease=disease, error=str(e))
            return cls.POPULATION_PREVALENCE.get(disease, 0.1)
    
    @classmethod
    def _calculate_age_factor(cls, age_of_onset: Optional[int], disease: str) -> float:
        """Calculate age factor for genetic risk (earlier onset = higher genetic component)."""
        
        if not age_of_onset:
            return 1.0  # Default factor if age unknown
        
        # Disease-specific typical onset ages
        typical_onset = {
            "diabetes": 45,
            "heart_disease": 55,
            "hypertension": 50,
            "cancer": 60,
            "stroke": 65,
            "alzheimers": 75,
            "depression": 30,
            "osteoporosis": 65,
            "asthma": 10,
            "obesity": 25
        }
        
        typical_age = typical_onset.get(disease, 50)
        
        # Earlier onset increases genetic factor
        if age_of_onset < typical_age:
            factor = 1.0 + (typical_age - age_of_onset) / typical_age * 0.5
        else:
            factor = max(0.5, 1.0 - (age_of_onset - typical_age) / typical_age * 0.3)
        
        return min(2.0, max(0.5, factor))
    
    @classmethod
    def _calculate_lifestyle_factor(cls, lifestyle_factors: Dict[str, str], disease: str) -> float:
        """Calculate how lifestyle factors modify genetic risk."""
        
        factor = 1.0
        
        # Smoking impact
        smoking = lifestyle_factors.get("smoking", "never")
        if smoking == "current":
            if disease in ["heart_disease", "stroke", "cancer"]:
                factor *= 1.3
        elif smoking == "former":
            if disease in ["heart_disease", "stroke", "cancer"]:
                factor *= 1.1
        
        # Exercise impact
        exercise = lifestyle_factors.get("exercise", "moderate")
        if exercise == "sedentary":
            if disease in ["diabetes", "heart_disease", "obesity"]:
                factor *= 1.2
        elif exercise == "active":
            if disease in ["diabetes", "heart_disease", "obesity"]:
                factor *= 0.8
        
        # Diet impact
        diet = lifestyle_factors.get("diet", "average")
        if diet == "poor":
            if disease in ["diabetes", "heart_disease", "obesity"]:
                factor *= 1.15
        elif diet == "excellent":
            if disease in ["diabetes", "heart_disease", "obesity"]:
                factor *= 0.85
        
        # Alcohol impact
        alcohol = lifestyle_factors.get("alcohol", "light")
        if alcohol == "heavy":
            if disease in ["heart_disease", "stroke", "cancer"]:
                factor *= 1.2
        
        return max(0.5, min(2.0, factor))
    
    @classmethod
    def _generate_tree_visualization(cls, family_members: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate family tree visualization data for D3.js."""
        
        nodes = [{"id": "user", "name": "You", "generation": 0, "x": 400, "y": 300, "type": "user"}]
        links = []
        
        # Position calculations
        generation_y = {0: 300, 1: 200, 2: 100, -1: 400}  # Y positions by generation
        generation_counts = defaultdict(int)
        
        for member in family_members:
            relation = member.get("relation", "")
            member_id = member.get("id", "")
            name = member.get("name", relation.replace("_", " ").title())
            
            # Determine generation
            generation = cls._get_generation(relation)
            generation_counts[generation] += 1
            
            # Calculate position
            base_x = 400
            if generation == 1:  # Parents
                x = base_x + (generation_counts[generation] - 1.5) * 200
            elif generation == 2:  # Grandparents
                x = base_x + (generation_counts[generation] - 2.5) * 150
            elif generation == -1:  # Children
                x = base_x + (generation_counts[generation] - 1.5) * 200
            else:
                x = base_x + (generation_counts[generation] - 1) * 100
            
            y = generation_y.get(generation, 300)
            
            # Add node
            node = {
                "id": member_id,
                "name": name,
                "relation": relation,
                "generation": generation,
                "x": x,
                "y": y,
                "type": "family_member",
                "health_conditions": member.get("health_conditions", []),
                "birth_year": member.get("birth_year"),
                "death_year": member.get("death_year")
            }
            nodes.append(node)
            
            # Add link to user
            link_type = cls._get_link_type(relation)
            links.append({
                "source": "user",
                "target": member_id,
                "type": link_type,
                "relation": relation
            })
        
        return {
            "nodes": nodes,
            "links": links,
            "layout_config": {
                "width": 800,
                "height": 600,
                "node_radius": 25,
                "link_distance": 100,
                "generations": list(generation_y.keys())
            }
        }
    
    @classmethod
    def _get_generation(cls, relation: str) -> int:
        """Get generation number for family member."""
        
        generation_map = {
            "father": 1, "mother": 1,
            "grandfather_paternal": 2, "grandmother_paternal": 2,
            "grandfather_maternal": 2, "grandmother_maternal": 2,
            "uncle_paternal": 1, "aunt_paternal": 1,
            "uncle_maternal": 1, "aunt_maternal": 1,
            "brother": 0, "sister": 0, "cousin": 0, "spouse": 0,
            "son": -1, "daughter": -1
        }
        
        return generation_map.get(relation, 0)
    
    @classmethod
    def _get_link_type(cls, relation: str) -> str:
        """Get link type for visualization."""
        
        if relation in ["father", "mother"]:
            return "parent"
        elif relation in ["son", "daughter"]:
            return "child"
        elif relation in ["brother", "sister"]:
            return "sibling"
        elif relation == "spouse":
            return "spouse"
        elif "grandfather" in relation or "grandmother" in relation:
            return "grandparent"
        elif "uncle" in relation or "aunt" in relation:
            return "aunt_uncle"
        elif relation == "cousin":
            return "cousin"
        else:
            return "relative"
    
    @classmethod
    async def get_family_health_insights(cls, user_id: str) -> Dict[str, Any]:
        """Generate comprehensive family health insights."""
        
        try:
            # Get family graph
            family_graph = await DatabaseService.mongodb_find_one(
                "family_graph",
                {"user_id": user_id}
            )
            
            if not family_graph:
                return {"message": "No family health data available"}
            
            family_members = family_graph.get("family_members", [])
            inherited_risks = family_graph.get("inherited_risks", {})
            
            # Generate insights
            insights = {
                "risk_summary": cls._generate_risk_summary(inherited_risks),
                "family_patterns": cls._analyze_family_patterns(family_members),
                "genetic_insights": cls._generate_genetic_insights(family_members, inherited_risks),
                "recommendations": cls._generate_family_recommendations(inherited_risks),
                "risk_timeline": cls._generate_risk_timeline(family_members),
                "comparative_analysis": cls._compare_with_population(inherited_risks)
            }
            
            return insights
            
        except Exception as e:
            logger.error("Failed to generate family insights", user_id=user_id, error=str(e))
            return {"error": str(e)}
    
    @classmethod
    def _generate_risk_summary(cls, inherited_risks: Dict[str, float]) -> Dict[str, Any]:
        """Generate risk summary with categorization."""
        
        high_risk = []
        medium_risk = []
        low_risk = []
        
        for disease, risk in inherited_risks.items():
            if risk >= 0.6:
                high_risk.append({"disease": disease, "risk": risk})
            elif risk >= 0.3:
                medium_risk.append({"disease": disease, "risk": risk})
            else:
                low_risk.append({"disease": disease, "risk": risk})
        
        # Sort by risk level
        high_risk.sort(key=lambda x: x["risk"], reverse=True)
        medium_risk.sort(key=lambda x: x["risk"], reverse=True)
        
        return {
            "high_risk_diseases": high_risk,
            "medium_risk_diseases": medium_risk,
            "low_risk_diseases": low_risk,
            "total_diseases_assessed": len(inherited_risks),
            "highest_risk": max(inherited_risks.values()) if inherited_risks else 0,
            "average_risk": sum(inherited_risks.values()) / len(inherited_risks) if inherited_risks else 0
        }
    
    @classmethod
    def _analyze_family_patterns(cls, family_members: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in family health history."""
        
        # Disease frequency analysis
        disease_counts = defaultdict(int)
        age_patterns = defaultdict(list)
        generation_patterns = defaultdict(lambda: defaultdict(int))
        
        for member in family_members:
            relation = member.get("relation", "")
            generation = cls._get_generation(relation)
            conditions = member.get("health_conditions", [])
            age_of_onset = member.get("age_of_onset", {})
            
            for condition in conditions:
                disease_counts[condition] += 1
                generation_patterns[generation][condition] += 1
                
                if condition in age_of_onset:
                    age_patterns[condition].append(age_of_onset[condition])
        
        # Find most common diseases
        common_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate average onset ages
        avg_onset_ages = {}
        for disease, ages in age_patterns.items():
            if ages:
                avg_onset_ages[disease] = sum(ages) / len(ages)
        
        return {
            "most_common_diseases": [{"disease": d, "count": c} for d, c in common_diseases],
            "average_onset_ages": avg_onset_ages,
            "generation_patterns": dict(generation_patterns),
            "total_family_members": len(family_members),
            "diseases_in_family": len(disease_counts)
        }
    
    @classmethod
    def _generate_genetic_insights(cls, family_members: List[Dict[str, Any]], inherited_risks: Dict[str, float]) -> List[str]:
        """Generate genetic insights and explanations."""
        
        insights = []
        
        # High genetic risk insights
        high_genetic_risks = {k: v for k, v in inherited_risks.items() if v > 0.6}
        
        if high_genetic_risks:
            top_risk = max(high_genetic_risks.items(), key=lambda x: x[1])
            insights.append(
                f"Your highest genetic risk is for {top_risk[0]} ({top_risk[1]:.1%} risk), "
                f"which is {top_risk[1]/cls.POPULATION_PREVALENCE.get(top_risk[0], 0.1):.1f}x "
                f"higher than the general population."
            )
        
        # Multiple affected relatives
        disease_counts = defaultdict(int)
        for member in family_members:
            for condition in member.get("health_conditions", []):
                disease_counts[condition] += 1
        
        multiple_affected = {d: c for d, c in disease_counts.items() if c > 1}
        if multiple_affected:
            top_clustered = max(multiple_affected.items(), key=lambda x: x[1])
            insights.append(
                f"{top_clustered[0].title()} appears in {top_clustered[1]} family members, "
                f"suggesting a strong genetic component."
            )
        
        # Early onset patterns
        early_onset_diseases = []
        for member in family_members:
            age_of_onset = member.get("age_of_onset", {})
            for disease, age in age_of_onset.items():
                typical_age = {"diabetes": 45, "heart_disease": 55, "cancer": 60}.get(disease, 50)
                if age < typical_age - 10:
                    early_onset_diseases.append(disease)
        
        if early_onset_diseases:
            insights.append(
                f"Early onset of {', '.join(set(early_onset_diseases))} in your family "
                f"may indicate stronger genetic predisposition."
            )
        
        return insights
    
    @classmethod
    def _generate_family_recommendations(cls, inherited_risks: Dict[str, float]) -> List[str]:
        """Generate personalized recommendations based on family history."""
        
        recommendations = []
        
        # High-risk disease recommendations
        high_risk_diseases = {k: v for k, v in inherited_risks.items() if v > 0.5}
        
        for disease, risk in high_risk_diseases.items():
            if disease == "diabetes":
                recommendations.append(
                    "Consider annual diabetes screening and maintain healthy weight due to family history"
                )
            elif disease == "heart_disease":
                recommendations.append(
                    "Regular cardiovascular screening and heart-healthy lifestyle modifications recommended"
                )
            elif disease == "cancer":
                recommendations.append(
                    "Follow enhanced cancer screening guidelines and discuss genetic counseling with your doctor"
                )
            elif disease == "hypertension":
                recommendations.append(
                    "Monitor blood pressure regularly and maintain low-sodium diet"
                )
        
        # General genetic counseling recommendation
        if len(high_risk_diseases) > 2:
            recommendations.append(
                "Consider genetic counseling to better understand your inherited risks and prevention strategies"
            )
        
        return recommendations
    
    @classmethod
    def _generate_risk_timeline(cls, family_members: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate risk timeline showing when diseases typically manifest."""
        
        age_ranges = {
            "20-30": [],
            "30-40": [],
            "40-50": [],
            "50-60": [],
            "60-70": [],
            "70+": []
        }
        
        for member in family_members:
            age_of_onset = member.get("age_of_onset", {})
            for disease, age in age_of_onset.items():
                if age < 30:
                    age_ranges["20-30"].append(disease)
                elif age < 40:
                    age_ranges["30-40"].append(disease)
                elif age < 50:
                    age_ranges["40-50"].append(disease)
                elif age < 60:
                    age_ranges["50-60"].append(disease)
                elif age < 70:
                    age_ranges["60-70"].append(disease)
                else:
                    age_ranges["70+"].append(disease)
        
        return age_ranges
    
    @classmethod
    def _compare_with_population(cls, inherited_risks: Dict[str, float]) -> Dict[str, Any]:
        """Compare family risks with population averages."""
        
        comparisons = {}
        
        for disease, family_risk in inherited_risks.items():
            population_risk = cls.POPULATION_PREVALENCE.get(disease, 0.1)
            relative_risk = family_risk / population_risk if population_risk > 0 else 1.0
            
            comparisons[disease] = {
                "family_risk": family_risk,
                "population_risk": population_risk,
                "relative_risk": relative_risk,
                "risk_category": "higher" if relative_risk > 1.5 else "similar" if relative_risk > 0.8 else "lower"
            }
        
        return comparisons